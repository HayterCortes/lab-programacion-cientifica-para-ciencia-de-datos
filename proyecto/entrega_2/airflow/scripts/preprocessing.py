#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocessing.py

Script de preprocesamiento de datos para SodAI Drinks.

"""

import pandas as pd
import os
import glob 

def load_and_preprocess_data(trans_path, cust_path, prod_path, batch_dir_path):
    """
    Carga, concatena batches incrementales, fusiona y limpia los datos.
    """
    
    # --- 1. Carga de Datos Estáticos ---
    print("Cargando datos estáticos (clientes y productos)...")
    try:
        df_clientes = pd.read_parquet(cust_path)
        df_productos = pd.read_parquet(prod_path)
    except FileNotFoundError as e:
        print(f"Error: No se encontraron archivos estáticos. {e}")
        raise
        
    # --- 2. LÓGICA DE BATCH ---
    print("Iniciando lógica de carga de transacciones (base + batches)...")
    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"No se encontró el archivo de transacciones base en {trans_path}")
        
    # Cargar transacciones base
    df_base_trans = pd.read_parquet(trans_path)
    print(f"Transacciones base cargadas. Forma: {df_base_trans.shape}")
    
    lista_dataframes_trans = [df_base_trans]
    
    # Buscar y cargar nuevos batches
    if os.path.exists(batch_dir_path):
        # Buscar todos los archivos .parquet en el directorio de batches
        batch_files = glob.glob(os.path.join(batch_dir_path, "*.parquet"))
        
        if batch_files:
            print(f"Encontrados {len(batch_files)} archivos de batch. Cargando y concatenando...")
            for f in sorted(batch_files): # Ordenar por nombre asegura el orden cronológico
                print(f"Cargando batch: {os.path.basename(f)}")
                df_batch = pd.read_parquet(f)
                lista_dataframes_trans.append(df_batch)
        else:
            print("No se encontraron archivos de batch en el directorio.")
    else:
        print(f"Directorio de batches '{batch_dir_path}' no encontrado. Usando solo datos base.")

    # Combinar todos los dataframes de transacciones
    df_transacciones = pd.concat(lista_dataframes_trans, ignore_index=True)
    print(f"DataFrame de transacciones total (base + batches) creado. Forma: {df_transacciones.shape}")
    
    # --- 3. Agregación Diaria y Manejo de Devoluciones ---
    print("Agregando transacciones a nivel diario para netear compras y devoluciones...")
    df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date']).dt.date 
    
    df_trans_agg = df_transacciones.groupby(
        ['customer_id', 'product_id', 'purchase_date']
    ).agg(items_neto=('items', 'sum')).reset_index()
    
    df_compras_efectivas = df_trans_agg[df_trans_agg['items_neto'] > 0].copy()
    
    df_compras_efectivas.rename(columns={'items_neto': 'items'}, inplace=True)

    # --- 4. Extracción de Componentes Temporales ---
    df_compras_efectivas['purchase_date'] = pd.to_datetime(df_compras_efectivas['purchase_date'])
    df_compras_efectivas['year'] = df_compras_efectivas['purchase_date'].dt.isocalendar().year
    df_compras_efectivas['week'] = df_compras_efectivas['purchase_date'].dt.isocalendar().week
    df_compras_efectivas['year_week'] = df_compras_efectivas['year'].astype(str) + '-W' + \
                                      df_compras_efectivas['week'].astype(str).str.zfill(2)

    # --- 5. Cruce de Información (Merge) ---
    print("Cruzando transacciones con datos de clientes y productos...")
    df_merged = pd.merge(df_compras_efectivas, df_clientes, on='customer_id', how='left')
    df_merged = pd.merge(df_merged, df_productos, on='product_id', how='left')
    
    # Manejar nulos en caso de que un batch introduzca un cliente/producto que no esté en los archivos maestros
    df_merged.dropna(subset=['customer_type', 'brand'], inplace=True)
    
    print(f"Datos preprocesados listos. Dimensiones finales: {df_merged.shape}")
    
    return df_merged