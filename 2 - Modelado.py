# Databricks notebook source
# MAGIC %md
# MAGIC # 🎯 Laboratorio — Fase 2: Modelado del Future Value (FV)
# MAGIC
# MAGIC ## Objetivo
# MAGIC Entrenar un modelo de **Future Value 30 días** para jugadores de lotería/juegos, usando:
# MAGIC - La tabla `gold_player_snapshot_features`
# MAGIC - División temporal de train/test
# MAGIC - Modelo **XGBoostRegressor**
# MAGIC - Registro y tracking con **MLflow**
# MAGIC
# MAGIC El resultado será un modelo productizable en la Fase 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración inicial
# MAGIC - Importamos librerías
# MAGIC - Cargamos la tabla Gold
# MAGIC - Exploramos el dataset

# COMMAND ----------

#!pip install xgboost==1.6.2

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

spark.sql("USE CATALOG future_value_lab")
spark.sql("USE SCHEMA gaming_poc")

df = spark.table("gold_player_snapshot_features").toPandas()
df = df[df["future_ggr_30d"].notna()]

print("Shape:", df.shape)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Definir el Target y Features
# MAGIC
# MAGIC Target:
# MAGIC - `future_ggr_30d`
# MAGIC
# MAGIC Features:
# MAGIC - Todas las demás columnas numéricas
# MAGIC - Excluimos: `player_id`, `snapshot_date`

# COMMAND ----------

TARGET = "future_ggr_30d"

drop_cols = ["player_id", "snapshot_date", TARGET]

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df[TARGET]

print("Features:", len(feature_cols))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. División Temporal Train / Test
# MAGIC
# MAGIC 🕒 Muy importante en modelos predictivos en gaming:
# MAGIC - Entrenar con snapshots antiguos
# MAGIC - Probar con snapshots recientes
# MAGIC
# MAGIC Para el POC:
# MAGIC - Train → primeras 70% fechas
# MAGIC - Test → últimas 30%

# COMMAND ----------

df_sorted = df.sort_values("snapshot_date")

cutoff = int(len(df_sorted) * 0.7)

train_df = df_sorted.iloc[:cutoff]
test_df  = df_sorted.iloc[cutoff:]

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_test,  y_test  = test_df[feature_cols],  test_df[TARGET]

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entrenar XGBoostRegressor con MLflow tracking
# MAGIC
# MAGIC Haremos:
# MAGIC - Run de MLflow
# MAGIC - Entrenamiento
# MAGIC - Logging de parámetros
# MAGIC - Logging de métricas MAE, RMSE
# MAGIC - Registro del modelo

# COMMAND ----------

mlflow.set_experiment("/Shared/gaming_fv_experiment")

with mlflow.start_run() as run:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Entrenamiento
    model.fit(X_train, y_train)

    # Predicciones
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    # Métricas
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test  = mean_absolute_error(y_test, pred_test)

    rmse_train = mean_squared_error(y_train, pred_train, squared=False)
    rmse_test  = mean_squared_error(y_test, pred_test, squared=False)

    # 👉 Firma del modelo (obligatoria para UC)
    signature = infer_signature(X_train, pred_train)

    # Logging de params y métricas
    mlflow.log_param("model_type", "xgboost_regressor")
    mlflow.log_param("horizon", "30 days")

    mlflow.log_metric("mae_train", mae_train)
    mlflow.log_metric("mae_test", mae_test)
    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_test", rmse_test)

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="future_value_model",     # nombre del artefacto dentro del run
        signature=signature,
        input_example=X_train.head(5)
    )

    run_id = run.info.run_id
    logged_model_uri = f"runs:/{run_id}/future_value_model"

print("Run ID:", run_id)
print("Logged model URI:", logged_model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Registrar Modelo en MLflow Model Registry
# MAGIC
# MAGIC Opcional pero recomendado.  
# MAGIC Usa el run ID generado arriba.

# COMMAND ----------

model_name = "gaming_future_value_model"

result = mlflow.register_model(
    f"runs:/{run_id}/future_value_model",
    model_name
)

result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Interpretabilidad: Feature Importance
# MAGIC
# MAGIC Esto ayuda a explicar a negocio **qué impulsa el Future Value**.
# MAGIC
# MAGIC XGBoost ya viene con `.feature_importances_`.

# COMMAND ----------

import matplotlib.pyplot as plt

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(imp_df["feature"][:20], imp_df["importance"][:20])
plt.title("Top 20 Feature Importances — XGBoost")
plt.gca().invert_yaxis()
plt.show()

imp_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. (Opcional) Interpretación con SHAP
# MAGIC
# MAGIC SHAP permite visualizar:
# MAGIC - Cómo cada feature afecta la predicción
# MAGIC - Efectos no lineales
# MAGIC
# MAGIC Para POC está bien usar force plots o summary plots.

# COMMAND ----------

# Uncomment si quieres usar SHAP
# import shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # 🎉 ¡Fin de la Fase 2!
# MAGIC
# MAGIC Ya tienes:
# MAGIC - Datos Gold → Features & Target
# MAGIC - Train/Test temporal
# MAGIC - Modelo XGBoost entrenado
# MAGIC - Métricas MAE y RMSE
# MAGIC - Modelo registrado en MLflow
# MAGIC - Interpretación con Feature Importance
# MAGIC
# MAGIC ## ¿Qué sigue?
# MAGIC **Fase 3 (Deployment)**:
# MAGIC - Crear pipeline de **batch scoring**
# MAGIC - Escribir tabla `gold.future_value_scores`
# MAGIC - Programar Job en Databricks
# MAGIC - (Opcional) Servir modelo vía Model Serving para tiempo real
