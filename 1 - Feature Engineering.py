# Databricks notebook source
# MAGIC %md
# MAGIC # 🧪 Laboratorio: Fase 1 - Construcción de Features y Target para Future Value Model (Gaming)
# MAGIC
# MAGIC En este laboratorio vamos a:
# MAGIC
# MAGIC 1. Cargar datos **bronze** (players, tickets, campañas).
# MAGIC 2. Crear tablas **silver** limpias.
# MAGIC 3. Construir métricas diarias por jugador.
# MAGIC 4. Calcular **features de los últimos 30/90 días**.
# MAGIC 5. Calcular el **target `future_ggr_30d`**.
# MAGIC 6. Materializar la tabla **`gold.player_snapshot_features`** lista para modelar.
# MAGIC
# MAGIC > Contexto: empresa de loterías/juegos con premios (lottery, scratch, sports).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración inicial
# MAGIC
# MAGIC En esta sección:
# MAGIC - Importamos librerías.
# MAGIC - Cargamos los CSV ficticios al volúmen creado en la siguiente celda
# MAGIC - Creamos las tablas `bronze.*` si aún no existen.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Configura tu CATALOGO si quieres aislar el laboratorio
spark.sql("CREATE CATALOG IF NOT EXISTS future_value_lab")
spark.sql("USE CATALOG future_value_lab")
spark.sql("CREATE SCHEMA IF NOT EXISTS gaming_poc")
spark.sql("USE SCHEMA gaming_poc")
spark.sql("CREATE VOLUME IF NOT EXISTS bronze_files")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.1 Cargar datasets ficticios (opcional)
# MAGIC Si ya tienes tablas `bronze.*` creadas desde otro flujo, **puedes saltarte esta celda**.

# COMMAND ----------

# Ajusta estas rutas a donde hayas subido tus CSV en Databricks
players_path  = "/Volumes/future_value_lab/gaming_poc/bronze_files/players.csv"          
tickets_path  = "/Volumes/future_value_lab/gaming_poc/bronze_files/tickets.csv"
campaign_path = "/Volumes/future_value_lab/gaming_poc/bronze_files/campaign_events.csv"

players_df = spark.read.option("header", True).csv(players_path, inferSchema=True)
tickets_df = spark.read.option("header", True).csv(tickets_path, inferSchema=True)
campaign_df = spark.read.option("header", True).csv(campaign_path, inferSchema=True)

# Crear tablas bronze
players_df.write.mode("overwrite").saveAsTable("bronze_players_raw")
tickets_df.write.mode("overwrite").saveAsTable("bronze_tickets_raw")
campaign_df.write.mode("overwrite").saveAsTable("bronze_campaign_events_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze → Silver
# MAGIC
# MAGIC Vamos a:
# MAGIC - Normalizar tipos de datos.
# MAGIC - Derivar columnas de fecha.
# MAGIC - Aplicar reglas de data quality básicas.
# MAGIC
# MAGIC Salidas:
# MAGIC - `silver.players`
# MAGIC - `silver.tickets`
# MAGIC - `silver.campaign_events`

# COMMAND ----------

# Crear schema "silver" lógico (las tablas se almacenan bajo gaming_poc.silver_*)
# No es obligatorio, pero ayuda a mantener orden.

# 1.1 Players
players_raw = spark.table("bronze_players_raw")

players_silver = (
    players_raw
    .withColumn("registration_ts", F.col("registration_ts").cast("timestamp"))
    .withColumn("registration_date", F.to_date("registration_ts"))
    .filter(F.col("player_id").isNotNull())
)

players_silver.write.mode("overwrite").format("delta").saveAsTable("silver_players")

# COMMAND ----------

# 1.2 Tickets
tickets_raw = spark.table("bronze_tickets_raw")

tickets_silver = (
    tickets_raw
    .withColumn("event_ts", F.col("event_ts").cast("timestamp"))
    .withColumn("event_date", F.to_date("event_ts"))
    .filter(F.col("player_id").isNotNull())
    .filter(F.col("stake") > 0)
    .filter(F.col("event_ts").isNotNull())
)

tickets_silver.write.mode("overwrite").format("delta").saveAsTable("silver_tickets")

# COMMAND ----------

# 1.3 Campaign events
campaign_raw = spark.table("bronze_campaign_events_raw")

campaign_silver = (
    campaign_raw
    .withColumn("event_ts", F.col("event_ts").cast("timestamp"))
    .withColumn("event_date", F.to_date("event_ts"))
    .filter(F.col("player_id").isNotNull())
)

campaign_silver.write.mode("overwrite").format("delta").saveAsTable("silver_campaign_events")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ **Checkpoint:**  
# MAGIC
# MAGIC Puedes explorar las tablas silver:
# MAGIC
# MAGIC ```sql
# MAGIC SELECT * FROM silver_players  LIMIT 5;
# MAGIC SELECT * FROM silver_tickets  LIMIT 5;
# MAGIC SELECT * FROM silver_campaign_events LIMIT 5;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Métricas diarias por jugador
# MAGIC
# MAGIC Ahora agregamos a nivel `(player_id, event_date)` para obtener:
# MAGIC - stake total diario
# MAGIC - premio total diario
# MAGIC - número de tickets
# MAGIC - diversidad de juegos
# MAGIC - ratios diarios por tipo de juego
# MAGIC
# MAGIC Salida:
# MAGIC - `silver.player_daily_metrics`

# COMMAND ----------

tickets = spark.table("silver_tickets")

player_daily = (
    tickets
    .groupBy("player_id", "event_date")
    .agg(
        F.sum("stake").alias("stake_sum_day"),
        F.sum("prize").alias("prize_sum_day"),
        F.count("*").alias("tickets_count_day"),
        F.countDistinct("game_type").alias("games_played_day"),
        # ratios por tipo de juego (usamos promedio de indicadores)
        F.avg(F.when(F.col("game_type") == "lottery", 1.0).otherwise(0.0)).alias("lottery_ratio_day"),
        F.avg(F.when(F.col("game_type") == "scratch", 1.0).otherwise(0.0)).alias("scratch_ratio_day"),
        F.avg(F.when(F.col("game_type") == "sports", 1.0).otherwise(0.0)).alias("sports_ratio_day")
    )
    .withColumn("ggr_day", F.col("stake_sum_day") - F.col("prize_sum_day"))
)

player_daily.write.mode("overwrite").format("delta").saveAsTable("silver_player_daily_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Features de los últimos 30 días por jugador (snapshots)
# MAGIC
# MAGIC Objetivo:
# MAGIC - Construir variables agregadas en ventana móvil de 30 días:
# MAGIC   - `stake_sum_30d`, `tickets_count_30d`, `ggr_sum_30d`
# MAGIC   - ratios 30d por tipo de juego
# MAGIC   - `active_days_30d`
# MAGIC   - `days_since_last_play`
# MAGIC
# MAGIC Estas serán nuestras **features históricas** para el modelo.

# COMMAND ----------

daily = spark.table("silver_player_daily_metrics")

# Ventana móvil de 30 "filas" (aprox 30 días con datos)
w_30d = (
    Window
    .partitionBy("player_id")
    .orderBy("event_date")
    .rowsBetween(-29, 0)
)

# Ventana para calcular lag (última fecha de juego)
w_lag = Window.partitionBy("player_id").orderBy("event_date")

snapshot_features = (
    daily
    .withColumn("stake_sum_30d", F.sum("stake_sum_day").over(w_30d))
    .withColumn("tickets_count_30d", F.sum("tickets_count_day").over(w_30d))
    .withColumn("ggr_sum_30d", F.sum("ggr_day").over(w_30d))
    .withColumn("lottery_ratio_30d", F.avg("lottery_ratio_day").over(w_30d))
    .withColumn("scratch_ratio_30d", F.avg("scratch_ratio_day").over(w_30d))
    .withColumn("sports_ratio_30d", F.avg("sports_ratio_day").over(w_30d))
    .withColumn("active_days_30d", F.count("event_date").over(w_30d))
    .withColumn("prev_event_date", F.lag("event_date").over(w_lag))
    .withColumn(
        "days_since_last_play",
        F.datediff("event_date", "prev_event_date")
    )
)

snapshot_features.createOrReplaceTempView("tmp_player_snapshot_features_base")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Enriquecer con información de jugadores y campañas
# MAGIC
# MAGIC Ahora:
# MAGIC - Añadimos **tenure** (`tenure_days`).
# MAGIC - Calculamos métricas de **engagement con campañas** en ventana de 90 días:
# MAGIC   - `campaigns_sent_90d`, `open_rate_90d`, `click_rate_90d`, `conv_rate_90d`.

# COMMAND ----------

players = spark.table("silver_players")

# 4.1 Añadir tenure del jugador
snap_with_player = (
    snapshot_features.alias("s")
    .join(players.alias("p"), "player_id", "left")
    .withColumn(
        "tenure_days",
        F.datediff(F.col("event_date"), F.col("registration_date"))
    )
)

# COMMAND ----------

# 4.2 Métricas diarias de campañas
campaign = spark.table("silver_campaign_events")

camp_daily = (
    campaign
    .groupBy("player_id", "event_date")
    .agg(
        F.sum(F.when(F.col("event_type") == "sent",    1).otherwise(0)).alias("campaigns_sent_day"),
        F.sum(F.when(F.col("event_type") == "open",    1).otherwise(0)).alias("campaigns_open_day"),
        F.sum(F.when(F.col("event_type") == "click",   1).otherwise(0)).alias("campaigns_click_day"),
        F.sum(F.when(F.col("event_type") == "convert", 1).otherwise(0)).alias("campaigns_convert_day"),
    )
)

camp_daily = camp_daily.withColumnRenamed("event_date", "campaign_event_date")
camp_daily = camp_daily.withColumnRenamed("player_id", "campaign_player_id")

snap_joined = (
    snap_with_player.alias("s")
    .join(
        camp_daily.alias("c"),
        on=[
            F.col("s.player_id") == F.col("campaign_player_id"),
            F.col("s.event_date") == F.col("campaign_event_date")
        ],
        how="left"
    )
    .na.fill(0, subset=["campaigns_sent_day", "campaigns_open_day", "campaigns_click_day", "campaigns_convert_day"])
)

# Ventana de 90 días para engagement
w_90d = (
    Window
    .partitionBy("s.player_id")
    .orderBy("s.event_date")
    .rowsBetween(-89, 0)
)

snap_with_engagement = (
    snap_joined
    .withColumn("campaigns_sent_90d",    F.sum("campaigns_sent_day").over(w_90d))
    .withColumn("campaigns_open_90d",    F.sum("campaigns_open_day").over(w_90d))
    .withColumn("campaigns_click_90d",   F.sum("campaigns_click_day").over(w_90d))
    .withColumn("campaigns_convert_90d", F.sum("campaigns_convert_day").over(w_90d))
    .withColumn(
        "open_rate_90d",
        F.when(F.col("campaigns_sent_90d") > 0,
               F.col("campaigns_open_90d") / F.col("campaigns_sent_90d"))
         .otherwise(F.lit(0.0))
    )
    .withColumn(
        "click_rate_90d",
        F.when(F.col("campaigns_sent_90d") > 0,
               F.col("campaigns_click_90d") / F.col("campaigns_sent_90d"))
         .otherwise(F.lit(0.0))
    )
    .withColumn(
        "conv_rate_90d",
        F.when(F.col("campaigns_sent_90d") > 0,
               F.col("campaigns_convert_90d") / F.col("campaigns_sent_90d"))
         .otherwise(F.lit(0.0))
    )
)

snap_with_engagement.createOrReplaceTempView("tmp_player_snapshot_with_engagement")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cálculo del target: `future_ggr_30d`
# MAGIC
# MAGIC Definimos el **Future Value** como:
# MAGIC
# MAGIC \- `future_ggr_30d = suma de GGR (stake - prize) de los próximos 30 días`  
# MAGIC
# MAGIC Usamos una ventana **hacia adelante** sobre `silver_player_daily_metrics`.

# COMMAND ----------

daily = spark.table("silver_player_daily_metrics")

w_future_30d = (
    Window
    .partitionBy("player_id")
    .orderBy("event_date")
    .rowsBetween(1, 30)   # siguientes 30 filas (≈ 30 días)
)

daily_with_future = (
    daily
    .withColumn("future_ggr_30d", F.sum("ggr_day").over(w_future_30d))
)

daily_with_future.createOrReplaceTempView("tmp_daily_with_future")

# COMMAND ----------

# Unir target futuro a los snapshots con features
snap_final = (
    snap_with_engagement.alias("s")
    .join(
        daily_with_future.select("player_id", "event_date", "future_ggr_30d").alias("f"),
        on=[
            F.col("s.player_id") == F.col("f.player_id"),
            F.col("s.event_date") == F.col("f.event_date")
        ],
        how="inner"   # Solo snapshots que tienen 30 días futuros disponibles
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Crear tabla Gold: `gold.player_snapshot_features`
# MAGIC
# MAGIC Esta tabla será la **base de entrenamiento** para el modelo de Future Value:
# MAGIC - Grain: `(player_id, snapshot_date)`
# MAGIC - Features: actividad 30d, engagement 90d, tenure…
# MAGIC - Target: `future_ggr_30d`.

# COMMAND ----------

gold_df = snap_final.select(
    F.col("s.player_id"),
    F.col("s.event_date").alias("snapshot_date"),
    # Features de actividad 30 días
    "stake_sum_30d",
    "tickets_count_30d",
    "ggr_sum_30d",
    "lottery_ratio_30d",
    "scratch_ratio_30d",
    "sports_ratio_30d",
    "active_days_30d",
    "days_since_last_play",
    # Tenure
    "tenure_days",
    # Engagement campañas (90 días)
    "campaigns_sent_90d",
    "campaigns_open_90d",
    "campaigns_click_90d",
    "campaigns_convert_90d",
    "open_rate_90d",
    "click_rate_90d",
    "conv_rate_90d",
    # Target
    "future_ggr_30d"
)

gold_df.write.mode("overwrite").format("delta").saveAsTable("gold_player_snapshot_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Fin de la Fase 1 (POC)
# MAGIC
# MAGIC Ya tienes:
# MAGIC
# MAGIC - `silver_player_daily_metrics` → actividad diaria por jugador  
# MAGIC - `gold_player_snapshot_features` → tabla de snapshots con:
# MAGIC   - Features históricas (30d / 90d / tenure)  
# MAGIC   - Target `future_ggr_30d`  
# MAGIC
# MAGIC Esta tabla está lista para usarse en la **Fase 2 (modelado)**, por ejemplo:
# MAGIC
# MAGIC ```python
# MAGIC df = spark.table("gold_player_snapshot_features")
# MAGIC display(df.limit(10))
# MAGIC ```
