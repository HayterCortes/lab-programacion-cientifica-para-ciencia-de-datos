"""
feature_engineering.py

Este script contiene todas las funciones para la ingeniería de características del proyecto SodAI Drinks, refactorizadas desde la Entrega 1.

"""

import pandas as pd
import numpy as np
from itertools import product
from typing import List, Dict, Any

# Importar la configuración centralizada
try:
    from .config import (
        NUMERIC_FEATURES, 
        CATEGORICAL_FEATURES, 
        ID_COLS, 
        TARGET_COL
    )
except ImportError:
    # Fallback si el script se ejecuta directamente
    print("Advertencia: No se pudo importar desde .config. Usando definiciones locales.")
    NUMERIC_FEATURES = [
        'lag_1_purchased', 'lag_1_items', 'lag_2_purchased', 'lag_2_items',
        'lag_3_purchased', 'lag_3_items', 'lag_4_purchased', 'lag_4_items',
        'rolling_4w_sum_purchased', 'rolling_4w_mean_purchased', 
        'rolling_4w_sum_items', 'rolling_4w_mean_items',
        'rolling_8w_sum_purchased', 'rolling_8w_mean_purchased', 
        'rolling_8w_sum_items', 'rolling_8w_mean_items',
        'week_of_year', 'month'
    ]
    CATEGORICAL_FEATURES = [
        'brand', 'package', 'sub_category', 'customer_type'
    ]
    ID_COLS = ['customer_id', 'product_id', 'target_week']
    TARGET_COL = 'target'


# --- 1. FUNCIÓN DE CREACIÓN DE BASE ---

def create_target_base(transactions_df: pd.DataFrame, target_weeks: list) -> pd.DataFrame:
    """
    Genera la tabla base (customer_id, product_id, target_week)
    y calcula el target (si hubo compra o no).
    """
    if not target_weeks:
        print("Advertencia: No se proporcionaron semanas objetivo (target_weeks).")
        return pd.DataFrame(columns=['customer_id', 'product_id', 'target_week', 'target'])

    print(f"Generando base para target_weeks: {target_weeks[0]} a {target_weeks[-1]}")

    target_transactions = transactions_df[
        transactions_df['year_week'].isin(target_weeks)
    ][['customer_id', 'product_id', 'year_week']].drop_duplicates()
    target_transactions['target'] = 1

    active_customers = transactions_df['customer_id'].unique()
    active_products = transactions_df['product_id'].unique()

    print(f"Clientes activos considerados: {len(active_customers)}")
    print(f"Productos activos considerados: {len(active_products)}")

    base_tuples = list(product(active_customers, active_products, target_weeks))
    df_base = pd.DataFrame(base_tuples, columns=['customer_id', 'product_id', 'target_week'])

    print(f"Tamaño base inicial (combinaciones): {df_base.shape}")

    # --- Optimización de Merge para Target ---
    # 1. Crear un MultiIndex para búsqueda rápida
    target_lookup = target_transactions.set_index(
        ['customer_id', 'product_id', 'year_week']
    )['target']
    
    # 2. Crear las tuplas de búsqueda desde df_base
    base_lookup_tuples = list(zip(
        df_base['customer_id'],
        df_base['product_id'],
        df_base['target_week']
    ))
    
    # 3. Usar .map() para asignar el target
    df_base['target'] = pd.Series(base_lookup_tuples).map(target_lookup).fillna(0).astype(int)
    
    print(f"Tamaño final base + target: {df_base.shape}")
    return df_base


# --- 2. FUNCIONES AUXILIARES DE FECHA Y LAGS ---

def year_week_to_date(yw_str: str) -> pd.Timestamp:
    """Convierte 'YYYY-WNN' a la fecha del lunes de esa semana ISO."""
    try:
        return pd.to_datetime(yw_str + '-1', format='%G-W%V-%u')
    except ValueError:
        print(f"Error convirtiendo: {yw_str}")
        return pd.NaT

def get_lag_week(target_week_str: str, lag: int) -> str:
    """Calcula la semana 'YYYY-WNN' correspondiente a 'lag' semanas antes."""
    target_date = year_week_to_date(target_week_str)
    if pd.isna(target_date):
        return None
    
    lag_date = target_date - pd.DateOffset(weeks=lag)
    
    iso_year = lag_date.isocalendar().year
    iso_week = lag_date.isocalendar().week
    return f"{iso_year}-W{str(iso_week).zfill(2)}"

def create_lag_maps(transactions_hist_agg_df: pd.DataFrame, 
                    feature_cols: List[str]) -> Dict[str, pd.Series]:
    """
    Crea los "diccionarios" de búsqueda (Series con MultiIndex) para todas las features de lag.
    """
    print("Creando mapas de búsqueda (diccionarios) para Lags...")
    hist_data_indexed = transactions_hist_agg_df.set_index(
        ['customer_id', 'product_id', 'year_week']
    )
    
    map_series_dict = {}
    for col in feature_cols:
        map_series_dict[col] = hist_data_indexed[col]
    
    print("Mapas de Lags creados.")
    return map_series_dict

def create_rolling_maps(transactions_hist_agg_df: pd.DataFrame, windows: List[int], 
                        feature_cols: List[str], agg_funcs: List[str]) -> Dict[str, pd.Series]:
    """
    Crea los "diccionarios" de búsqueda (Series con MultiIndex) para todas las features de rolling.
    """
    print("\nGenerando Features de Ventana Móvil (Cálculo)...")
    hist_data = transactions_hist_agg_df.copy()

    hist_data['date'] = hist_data['year_week'].apply(year_week_to_date)
    hist_data = hist_data.sort_values(by=['customer_id', 'product_id', 'date'])

    hist_data_indexed = hist_data.set_index('date')
    grouped = hist_data_indexed.groupby(['customer_id', 'product_id'], observed=True)
    
    df_rolling_features = None

    for window in windows:
        print(f"Calculando para ventana de {window} semanas...")
        for feature_col in feature_cols:
            for agg_func in agg_funcs:
                col_name = f'rolling_{window}w_{agg_func}_{feature_col}'
                print(f"Generando: {col_name}")
                
                rolling_col = grouped[feature_col].rolling(window=window, min_periods=1).agg(agg_func)
                rolling_col = rolling_col.reset_index()
                rolling_col = rolling_col.rename(columns={feature_col: col_name})
                
                rolling_col = pd.merge(
                    rolling_col, 
                    hist_data[['customer_id', 'product_id', 'date', 'year_week']].drop_duplicates(),
                    on=['customer_id', 'product_id', 'date'], 
                    how='left'
                )
                rolling_col = rolling_col.sort_values(by=['customer_id', 'product_id', 'year_week'])
                
                rolling_col[col_name] = rolling_col.groupby(
                    ['customer_id', 'product_id'], observed=True
                )[col_name].shift(1)
                
                rolling_col = rolling_col[['customer_id', 'product_id', 'year_week', col_name]].dropna(subset=['year_week'])
                
                # Juntar en un solo DataFrame
                if df_rolling_features is None:
                    df_rolling_features = rolling_col
                else:
                    df_rolling_features = df_rolling_features.merge(
                        rolling_col, on=['customer_id', 'product_id', 'year_week'], how='outer'
                    )

    # --- Crear los Mapas (diccionarios) ---
    print("Creando mapas de búsqueda (diccionarios) para Rolling Features...")
    df_rolling_features = df_rolling_features.rename(columns={'year_week': 'target_week'})
    df_rolling_indexed = df_rolling_features.set_index(['customer_id', 'product_id', 'target_week'])
    
    map_series_dict = {}
    for col in df_rolling_indexed.columns:
        map_series_dict[col] = df_rolling_indexed[col]

    print("Mapas de Rolling Features creados.")
    return map_series_dict


# --- 3. FUNCIÓN DE FEATURES TEMPORALES ---

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features temporales (semana del año, mes) a partir de target_week. 
    """
    df_temp = df.copy()
    df_temp['date'] = df_temp['target_week'].apply(year_week_to_date)
    df_temp['week_of_year'] = df_temp['date'].dt.isocalendar().week.astype(int)
    df_temp['month'] = df_temp['date'].dt.month.astype(int)
    df_temp = df_temp.drop(columns=['date'])
    return df_temp


# --- 4. FUNCIÓN MAESTRA ORQUESTADORA ---

def create_feature_matrix(df_merged: pd.DataFrame, target_weeks: List[str]) -> pd.DataFrame:
    """
    Función maestra para generar la matriz de features completa 
    para un conjunto de semanas objetivo dadas.
    Orquesta todas las funciones anteriores.
    """
    
    print(f"Iniciando creación de matriz de features para {len(target_weeks)} semanas.")
    
    # 1. Crear la base (combinaciones customer-product-week)
    df_base = create_target_base(df_merged, target_weeks)
    
    # 2. Refinar datos históricos (agregar por semana)
    hist_agg = df_merged.groupby(
        ['customer_id', 'product_id', 'year_week'], observed=True
    )['items'].sum().reset_index()
    hist_agg['purchased'] = 1
    
    # 3. Crear todos los mapas (diccionarios) de búsqueda de una vez
    lag_cols = ['purchased', 'items']
    lag_maps = create_lag_maps(hist_agg, lag_cols)
    
    rolling_windows = [4, 8]
    rolling_cols = ['purchased', 'items']
    rolling_aggs = ['sum', 'mean']
    rolling_maps = create_rolling_maps(hist_agg, rolling_windows, rolling_cols, rolling_aggs)
    
    # 4. Añadir Lags (Usando .map())
    lags_to_compute = [1, 2, 3, 4]
    
    # Crear mapa de lookup_week
    unique_target_weeks = df_base['target_week'].unique()
    lag_map_list = []
    for tw in unique_target_weeks:
        for lag in lags_to_compute:
            lag_map_list.append((tw, lag, get_lag_week(tw, lag)))
    df_lag_map = pd.DataFrame(lag_map_list, columns=['target_week', 'lag', 'lookup_week']).set_index('target_week')
    
    for lag in lags_to_compute:
        print(f"Aplicando lags de {lag} semana(s) (con .map())...")
        # Obtener las lookup_weeks para todas las filas
        lookup_week_series = df_base['target_week'].map(df_lag_map[df_lag_map['lag'] == lag]['lookup_week'])
        
        # Crear tuplas de búsqueda
        lookup_tuples = list(zip(
            df_base['customer_id'], 
            df_base['product_id'], 
            lookup_week_series
        ))
        
        for col in lag_cols:
            new_col_name = f'lag_{lag}_{col}'
            map_series = lag_maps[col]
            df_base[new_col_name] = pd.Series(lookup_tuples).map(map_series).fillna(0)

    # 5. Añadir Ventanas Móviles (Usando .map())
    print("Aplicando features de ventana móvil (con .map())...")
    # Crear tuplas de búsqueda para la base
    base_lookup_tuples = list(zip(
        df_base['customer_id'],
        df_base['product_id'],
        df_base['target_week']
    ))
    base_lookup_series = pd.Series(base_lookup_tuples)
    
    for col_name, map_series in rolling_maps.items():
        print(f"Aplicando: {col_name}")
        df_base[col_name] = base_lookup_series.map(map_series).fillna(0)
    
    # 6. Añadir Features Estáticas (Usando .map())
    print("Aplicando features estáticas (con .map())...")
    product_map = df_merged.drop_duplicates('product_id').set_index('product_id')
    client_map = df_merged.drop_duplicates('customer_id').set_index('customer_id')
    
    df_base['brand'] = df_base['product_id'].map(product_map['brand'])
    df_base['package'] = df_base['product_id'].map(product_map['package'])
    df_base['sub_category'] = df_base['product_id'].map(product_map['sub_category'])
    df_base['customer_type'] = df_base['customer_id'].map(client_map['customer_type'])
    
    # 7. Añadir Features Temporales
    print("Aplicando features temporales...")
    df_base = add_temporal_features(df_base)
    
    # 8. Limpieza final y verificación
    print("Limpiando y verificando tipos de datos...")
    for col in CATEGORICAL_FEATURES:
        if col in df_base.columns:
            df_base[col] = df_base[col].fillna("Desconocido").astype('category')
        else:
            print(f"Advertencia: Columna categórica esperada '{col}' no encontrada.")
    
    print(f"Matriz de features creada. Dimensiones finales: {df_base.shape}")
    
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    cols_finales = ID_COLS + all_features + [TARGET_COL]
    
    # Asegurarnos de que todas las columnas existen
    cols_faltantes = [c for c in cols_finales if c not in df_base.columns]
    if cols_faltantes:
        print(f"ADVERTENCIA: Faltan las siguientes columnas en el DataFrame final: {cols_faltantes}")
        for c in cols_faltantes:
            if c in CATEGORICAL_FEATURES:
                df_base[c] = "Desconocido"
                df_base[c] = df_base[c].astype('category')
            else:
                df_base[c] = 0

    return df_base[cols_finales]

# --- Bloque de prueba (opcional) ---
if __name__ == "__main__":
    print("Ejecutando feature_engineering.py como script independiente (modo de prueba)...")
    pass