"""
validation_drift.py

Este script contiene la lógica de detección de Data Drift
para el pipeline de SodAI Drinks.
"""

import pandas as pd
from scipy.stats import ks_2samp
from typing import List

def check_data_drift(
    new_data_path: str, 
    reference_data_path: str, 
    features_to_monitor: List[str], 
    p_value_threshold: float = 0.05
) -> bool:
    """
    Compara un nuevo conjunto de datos con un conjunto de referencia
    usando el test de Kolmogorov-Smirnov (K-S) en las features numéricas.

    Args:
        new_data_path (str): Ruta al Parquet con los datos completos (base + batches).
        reference_data_path (str): Ruta al Parquet con los datos de referencia (solo base).
        features_to_monitor (List[str]): Lista de columnas numéricas a monitorear.
        p_value_threshold (float): El umbral de p-value para declarar "drift".

    Returns:
        bool: True si se detecta drift, False en caso contrario.
    """
    
    print(f"Iniciando Detección de Drift. Comparando '{new_data_path}' contra '{reference_data_path}'")
    print(f"Monitoreando {len(features_to_monitor)} features.")

    try:
        # Cargar solo las columnas necesarias para eficiencia
        df_new = pd.read_parquet(new_data_path, columns=features_to_monitor)
        df_ref = pd.read_parquet(reference_data_path, columns=features_to_monitor)
    except Exception as e:
        print(f"Error al cargar los dataframes para la detección de drift: {e}")
        print("Asumiendo DRIFT por precaución y gatillando re-entrenamiento.")
        return True # Si no podemos comparar, es más seguro re-entrenar

    drift_detected_count = 0
    
    for col in features_to_monitor:
        if col not in df_new.columns or col not in df_ref.columns:
            print(f"Advertencia: Columna '{col}' no encontrada en uno de los dataframes. Saltando chequeo.")
            continue
            
        # Ejecutar el Test K-S de 2 muestras
        # Compara si las dos distribuciones son diferentes
        try:
            ks_statistic, p_value = ks_2samp(df_new[col], df_ref[col])
            
            if p_value < p_value_threshold:
                print(f"¡DRIFT DETECTADO! Feature: '{col}' (p-value: {p_value:.4f} < {p_value_threshold})")
                drift_detected_count += 1
            else:
                print(f"Feature '{col}' estable. (p-value: {p_value:.4f})")
                
        except Exception as e:
            print(f"Error al procesar el test K-S para la columna '{col}': {e}")
            # Considerar un error en el test como una señal de drift
            drift_detected_count += 1

    if drift_detected_count > 0:
        print(f"Resultado: Drift detectado en {drift_detected_count} feature(s). Se recomienda re-entrenar.")
        return True
    else:
        print("Resultado: Datos estables. No se detectó drift significativo.")
        return False