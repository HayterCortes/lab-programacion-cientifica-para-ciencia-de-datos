"""
model_prediction.py

Este script contiene la lógica para generar las predicciones semanales
del proyecto SodAI Drinks, como se especifica en la Entrega 2.

"""

import pandas as pd
import mlflow
import mlflow.sklearn
import os
import sys
from typing import List

# --- 1. IMPORTAR MÓDULOS DEL PAQUETE ---
try:
    from . import feature_engineering
    from .config import ID_COLS, TARGET_COL
except ImportError:
    print("Advertencia: No se pudo importar desde .config y .feature_engineering. Intentando importación absoluta.")
    try:
        import feature_engineering
        from config import ID_COLS, TARGET_COL
    except ImportError as e:
        print(f"Error: No se pudieron importar las dependencias.")
        print(e)
        sys.exit(1)


# --- 2. FUNCIÓN AUXILIAR DE FECHA ---

def get_next_week(latest_week_str: str) -> str:
    """
    Calcula la semana 'YYYY-WNN' siguiente a la semana dada.
    """
    latest_date = feature_engineering.year_week_to_date(latest_week_str)
    if pd.isna(latest_date):
        raise ValueError(f"Formato de semana inválido: {latest_week_str}")
    
    next_week_date = latest_date + pd.DateOffset(weeks=1)
    
    iso_year = next_week_date.isocalendar().year
    iso_week = next_week_date.isocalendar().week
    return f"{iso_year}-W{str(iso_week).zfill(2)}"


# --- 3. FUNCIÓN PRINCIPAL DE PREDICCIÓN ---

def generate_predictions(
    historical_data: pd.DataFrame, 
    model_uri: str,
    output_path: str,
    mlflow_tracking_uri: str
) -> None:
    """
    Orquesta el proceso completo de generación de predicciones.
    """
    
    print("--- Iniciando Proceso de Generación de Predicciones ---")

    # --- 1. Conectar al Servidor MLflow ---
    import mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # ------------------------------------

    # --- 2. Determinar Semana Objetivo ---
    try:
        latest_week = historical_data['year_week'].max()
    except KeyError:
        print("Error: La columna 'year_week' no se encuentra en los datos históricos.")
        raise
        
    target_week_str = get_next_week(latest_week)
    target_weeks_list = [target_week_str]
    
    print(f"Datos históricos encontrados hasta: {latest_week}")
    print(f"Generando predicciones para la semana objetivo: {target_week_str}")

    # --- 3. Cargar Modelo desde MLflow ---
    print(f"Cargando modelo desde MLflow URI: {model_uri} ...")
    try:
        model_pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error al cargar el modelo desde MLflow: {e}")
        raise
    print("Modelo cargado exitosamente.")

    # --- 4. Generar Matriz de Features para la Semana Objetivo ---
    print("Iniciando generación de matriz de features para la semana objetivo...")
    df_features_to_predict = feature_engineering.create_feature_matrix(
        df_merged=historical_data, 
        target_weeks=target_weeks_list
    )
    print("Matriz de features generada.")

    # --- 5. Realizar Predicciones ---
    X_to_predict = df_features_to_predict.drop(columns=[TARGET_COL] + ID_COLS)
    
    print(f"Generando predicciones para {len(X_to_predict)} combinaciones cliente-producto...")
    probabilities = model_pipeline.predict_proba(X_to_predict)
    
    prob_compra = probabilities[:, 1]
    print("Predicciones generadas.")

    # --- 6. Formatear y Guardar Salida ---
    df_results = df_features_to_predict[ID_COLS].copy()
    df_results['probability'] = prob_compra
    
    df_results_sorted = df_results.sort_values(
        by=['customer_id', 'probability'], 
        ascending=[True, False]
    )
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Predicciones guardadas exitosamente en: {output_path}")
    df_results_sorted.to_csv(output_path, index=False)
    
    print("--- Proceso de Generación de Predicciones Finalizado ---")


# --- Bloque de prueba (opcional) ---
if __name__ == "__main__":
    print("Ejecutando model_prediction.py como script independiente (modo de prueba)...")
    pass