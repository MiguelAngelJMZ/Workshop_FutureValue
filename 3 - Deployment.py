# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Laboratorio — Fase 3: Deployment & Batch Scoring del Future Value Model
# MAGIC
# MAGIC ## Objetivo
# MAGIC En este laboratorio vamos a:
# MAGIC
# MAGIC 1. Cargar el modelo de **Future Value 30d** desde MLflow / Unity Catalog.
# MAGIC 2. Seleccionar un conjunto de snapshots a predecir (por fecha o rango).
# MAGIC 3. Ejecutar **batch scoring** sobre `gold_player_snapshot_features`.
# MAGIC 4. Guardar los resultados en la tabla `gold_player_future_value_scores`.
# MAGIC 5. Dejar el notebook listo para ser programado como **Job diario/semanal**.
# MAGIC
# MAGIC Asumimos que:
# MAGIC - La Fase 1 y 2 ya corrieron.
# MAGIC - Existe la tabla `gaming_poc.gold_player_snapshot_features`.
# MAGIC - Tienes un modelo registrado en MLflow Model Registry (UC opcional).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración y parámetros
# MAGIC
# MAGIC Aquí definimos:
# MAGIC - Base de datos de trabajo.
# MAGIC - Nombre del modelo en el Registry.
# MAGIC - Alias o versión del modelo.
# MAGIC - Rango de fechas a escorar.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

import mlflow
import mlflow.pyfunc

spark.sql("USE CATALOG future_value_lab")
spark.sql("USE SCHEMA gaming_poc")

MODEL_NAME = "future_value_lab.gaming_poc.gaming_future_value_model" 

ALIAS = "champion"         

# Modelo URI estilo MLflow 3
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"

print("Model URI:", MODEL_URI)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Seleccionar snapshots a escorar
# MAGIC
# MAGIC En un escenario real:
# MAGIC - Este notebook se correría diario/semanal.
# MAGIC - Escorarías:
# MAGIC   - El último `snapshot_date` disponible, o
# MAGIC   - Un rango de fechas reciente.
# MAGIC
# MAGIC Para el laboratorio:
# MAGIC - Tomaremos por defecto el **último snapshot_date**.

# COMMAND ----------

snapshots_df = spark.table("gold_player_snapshot_features")

# Obtener la última fecha de snapshot disponible
max_date = snapshots_df.agg(F.max("snapshot_date").alias("max_date")).collect()[0]["max_date"]
print("Último snapshot_date disponible:", max_date)

# Si quieres fijar una fecha manualmente, puedes sobreescribir:
# target_snapshot_date = F.to_date(F.lit("2024-10-01")).cast("date")

target_snapshot_date = max_date

to_score_df = snapshots_df.filter(F.col("snapshot_date") == target_snapshot_date)

print("Número de jugadores a escorar:", to_score_df.count())

display(to_score_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Definir columnas de features para scoring
# MAGIC
# MAGIC Usamos la misma lógica de la Fase 2:
# MAGIC - Target de entrenamiento: `future_ggr_30d`
# MAGIC - Features = todas las demás columnas numéricas, excluyendo:
# MAGIC   - `player_id`
# MAGIC   - `snapshot_date`
# MAGIC   - `future_ggr_30d`
# MAGIC
# MAGIC 🔎 En producción normalmente consolidarías esta lógica en un módulo compartido, pero aquí lo dejamos explícito para el lab.

# COMMAND ----------

all_cols = to_score_df.columns

TARGET_COL = "future_ggr_30d"
ID_COLS = ["player_id", "snapshot_date"]

feature_cols = [c for c in all_cols if c not in ID_COLS + [TARGET_COL]]

print("Features usadas para scoring:")
print(feature_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cargar el modelo desde MLflow / Unity Catalog
# MAGIC
# MAGIC Usaremos `mlflow.pyfunc.spark_udf` para:
# MAGIC - Cargar el modelo como UDF de Spark.
# MAGIC - Aplicarlo sobre las columnas de features.
# MAGIC
# MAGIC Esto permite que el scoring escale de forma distribuida en el cluster.

# COMMAND ----------

from pyspark.sql.functions import col

# Crear UDF de predicción
predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=MODEL_URI,
    result_type=DoubleType(),
    #env_manager="conda"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ejecutar batch scoring
# MAGIC
# MAGIC Añadimos una columna:
# MAGIC - `fv_30d_pred` = predicción del modelo para el Future Value 30 días.
# MAGIC
# MAGIC Después calcularemos:
# MAGIC - Ranking por jugador dentro del snapshot.
# MAGIC - Percentil / decil de valor.

# COMMAND ----------

scored_df = (
    to_score_df
    .withColumn(
        "fv_30d_pred",
        predict_udf(*[F.col(c) for c in feature_cols])
    )
)

display(scored_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Añadir ranking, percentil y deciles
# MAGIC
# MAGIC Estas columnas son muy útiles para:
# MAGIC - Segmentación (Top 10%, 20%, etc.)
# MAGIC - Dashboards
# MAGIC - Campañas de marketing
# MAGIC
# MAGIC Calculamos:
# MAGIC - `fv_rank` → ranking descendente por `fv_30d_pred`
# MAGIC - `fv_percentile` → `percent_rank` de Spark
# MAGIC - `fv_decile` → 1 a 10 según percentil

# COMMAND ----------

w_rank = (
    Window
    .partitionBy("snapshot_date")
    .orderBy(F.col("fv_30d_pred").desc())
)

scored_ranked_df = (
    scored_df
    .withColumn("fv_rank", F.row_number().over(w_rank))
    .withColumn("fv_percentile", F.percent_rank().over(w_rank))
    .withColumn(
        "fv_decile",
        (F.col("fv_percentile") * 10).cast("int") + 1
    )
)

display(scored_ranked_df.select("player_id", "snapshot_date", "fv_30d_pred", "fv_decile").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Escribir resultados en `gold_player_future_value_scores`
# MAGIC
# MAGIC Creamos (si no existe) una tabla Gold de scores:
# MAGIC
# MAGIC ```text
# MAGIC gold_player_future_value_scores:
# MAGIC   player_id
# MAGIC   snapshot_date
# MAGIC   fv_30d_pred
# MAGIC   fv_rank
# MAGIC   fv_percentile
# MAGIC   fv_decile
# MAGIC   scored_at
# MAGIC ```
# MAGIC
# MAGIC Usaremos un **MERGE** para:
# MAGIC - Actualizar scores si ya existían para ese `snapshot_date`.
# MAGIC - Insertar si son nuevos.

# COMMAND ----------

# Crear tabla si no existe
spark.sql("""
CREATE TABLE IF NOT EXISTS gold_player_future_value_scores (
  player_id        STRING,
  snapshot_date    DATE,
  fv_30d_pred      DOUBLE,
  fv_rank          BIGINT,
  fv_percentile    DOUBLE,
  fv_decile        INT,
  scored_at        TIMESTAMP
)
USING delta
""")

# Preparar DF a escribir
scores_to_write = (
    scored_ranked_df
    .select(
        "player_id",
        "snapshot_date",
        "fv_30d_pred",
        "fv_rank",
        "fv_percentile",
        "fv_decile"
    )
    .withColumn("scored_at", F.current_timestamp())
)

scores_to_write.createOrReplaceTempView("scores_to_write")

# MERGE por player_id + snapshot_date
spark.sql("""
MERGE INTO gold_player_future_value_scores AS t
USING scores_to_write AS s
ON  t.player_id     = s.player_id
AND t.snapshot_date = s.snapshot_date
WHEN MATCHED THEN UPDATE SET
  t.fv_30d_pred   = s.fv_30d_pred,
  t.fv_rank       = s.fv_rank,
  t.fv_percentile = s.fv_percentile,
  t.fv_decile     = s.fv_decile,
  t.scored_at     = s.scored_at
WHEN NOT MATCHED THEN INSERT (
  player_id,
  snapshot_date,
  fv_30d_pred,
  fv_rank,
  fv_percentile,
  fv_decile,
  scored_at
) VALUES (
  s.player_id,
  s.snapshot_date,
  s.fv_30d_pred,
  s.fv_rank,
  s.fv_percentile,
  s.fv_decile,
  s.scored_at
)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ **Checkpoint: revisar scores escritos**
# MAGIC
# MAGIC Podemos ver algunos registros de la tabla final:
# MAGIC
# MAGIC ```sql
# MAGIC SELECT * 
# MAGIC FROM gold_player_future_value_scores
# MAGIC ORDER BY snapshot_date DESC, fv_30d_pred DESC
# MAGIC LIMIT 20;
# MAGIC ```

# COMMAND ----------

display(
    spark.sql("""
    SELECT * 
    FROM gold_player_future_value_scores
    ORDER BY snapshot_date DESC, fv_30d_pred DESC
    LIMIT 20
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cómo programar este notebook como Job 💼
# MAGIC
# MAGIC En un escenario real:
# MAGIC - Este notebook se configura como **Job en Databricks**.
# MAGIC - Frecuencia típica:
# MAGIC   - Diario (si tus snapshots son diarios).
# MAGIC   - Semanal (si trabajas por semanas).
# MAGIC
# MAGIC Pasos (manual, vía UI):
# MAGIC 1. En el menú izquierdo ve a **Jobs & Pipelines → Jobs**.
# MAGIC 2. Clic en **Create Job**.
# MAGIC 3. Selecciona este notebook `Fase_3_Deployment_Scoring`.
# MAGIC 4. Configura el **cluster** (puede ser un Job Cluster pequeño para el POC).
# MAGIC 5. Define el **schedule** (por ejemplo, diario a las 7:00 AM).
# MAGIC 6. Guarda y prueba corriendo un **Run Now**.
# MAGIC
# MAGIC A partir de ese momento:
# MAGIC - La tabla `gold_player_future_value_scores` se irá llenando automáticamente.
# MAGIC - Podrás consumirla desde:
# MAGIC   - Dashboards (Lakeview).
# MAGIC   - Analistas de negocio.

# COMMAND ----------

# MAGIC %md
# MAGIC # 🎉 Fin de la Fase 3
# MAGIC
# MAGIC Ya tienes:
# MAGIC
# MAGIC - ✅ Un modelo registrado en MLflow / UC (Fase 2).  
# MAGIC - ✅ Un pipeline de **batch scoring** que:
# MAGIC   - Lee `gold_player_snapshot_features`.
# MAGIC   - Carga el modelo desde `models:/...`.
# MAGIC   - Escribe `gold_player_future_value_scores` con predicciones + ranking.  
# MAGIC - ✅ Un notebook listo para ser usado como **Job recurrente**.
