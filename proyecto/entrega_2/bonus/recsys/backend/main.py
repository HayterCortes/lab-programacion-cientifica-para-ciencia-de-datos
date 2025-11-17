"""
Backend (FastAPI) para el Sistema de Recomendación.

Implementa un filtro colaborativo item-item basado en Coseno de Similitud.
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuración ---
#
# El Dockerfile montará el archivo de transacciones aquí
DATA_FILE_PATH = "/app/data/transacciones.parquet"

# Variable global para mantener los "modelos" (matrices) en memoria
recsys_models = {
    "user_item_matrix": None,
    "item_sim_matrix": None
}

# --- Funciones de "Entrenamiento" (Construcción de Matriz) ---

def build_similarity_matrix():
    """
    Carga las transacciones y construye las matrices de
    usuario-item y de similitud item-item.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ADVERTENCIA: No se encontró {DATA_FILE_PATH}. El RecSys no funcionará.")
        return None, None

    print("Cargando transacciones para el sistema de recomendación...")
    df_trans = pd.read_parquet(DATA_FILE_PATH)

    # 1. Crear matriz de interacciones (solo nos importa si compró o no)
    print("Creando matriz de interacciones Usuario-Item...")
    df_interactions = df_trans[['customer_id', 'product_id']].drop_duplicates()

    # Usar pd.crosstab para crear la matriz binaria
    user_item_matrix = pd.crosstab(
        df_interactions['customer_id'], 
        df_interactions['product_id']
    )
    print(f"Matriz Usuario-Item creada. Forma: {user_item_matrix.shape}")

    # 2. Crear matriz de similitud Item-Item
    print("Calculando similitud Coseno Item-Item...")
    # Transponemos la matriz (item x user) y calculamos la similitud
    item_sim_matrix_raw = cosine_similarity(user_item_matrix.T)

    # Convertir de nuevo a un DataFrame de Pandas con etiquetas
    item_sim_matrix = pd.DataFrame(
        item_sim_matrix_raw,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    print(f"Matriz de Similitud Item-Item creada. Forma: {item_sim_matrix.shape}")

    return user_item_matrix, item_sim_matrix

# --- Eventos de Ciclo de Vida de la App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga las matrices de similitud al iniciar la aplicación.
    """
    print("Iniciando aplicación RecSys...")
    user_item, item_sim = build_similarity_matrix()
    recsys_models["user_item_matrix"] = user_item
    recsys_models["item_sim_matrix"] = item_sim
    yield
    # Limpieza
    print("Apagando aplicación RecSys...")
    recsys_models.clear()

# --- Creación de la App ---

app = FastAPI(
    title="SodAI Drinks - RecSys (Bonus)",
    description="API de Filtro Colaborativo Item-Item",
    version="1.0.0",
    lifespan=lifespan
)

# --- Endpoints de la API ---

@app.get("/")
def read_root():
    return {"status": "RecSys (Bonus) API está viva."}

@app.get("/recommendations/{customer_id}")
def get_recommendations(customer_id: int):
    """
    Genera 5 recomendaciones de Filtro Colaborativo
    para un ID de cliente específico.
    """
    user_item_matrix = recsys_models["user_item_matrix"]
    item_sim_matrix = recsys_models["item_sim_matrix"]

    if user_item_matrix is None or item_sim_matrix is None:
        raise HTTPException(status_code=503, 
                            detail="Los modelos de recomendación no están cargados.")

    if customer_id not in user_item_matrix.index:
        raise HTTPException(status_code=404, 
                            detail=f"customer_id {customer_id} no encontrado en los datos de transacciones.")

    # 1. Obtener los productos que el cliente si ha comprado
    bought_items_series = user_item_matrix.loc[customer_id]
    bought_items = bought_items_series[bought_items_series > 0].index

    if len(bought_items) == 0:
        raise HTTPException(status_code=404, 
                            detail=f"El cliente {customer_id} no tiene compras registradas.")

    # 2. Calcular el puntaje total de similitud
    # Suma las similitudes de todos los items que compró
    all_similar_items = item_sim_matrix[bought_items].sum(axis=1)

    # 3. Filtrar los productos que ya ha comprado
    all_similar_items = all_similar_items.drop(labels=bought_items)

    # 4. Obtener el Top 5
    top_5_recs = all_similar_items.nlargest(5)

    # Formatear como JSON
    recommendations = [
        {"product_id": int(product), "similarity_score": round(score, 4)}
        for product, score in top_5_recs.items()
    ]

    return {
        "customer_id": customer_id,
        "based_on_purchases": bought_items.tolist(),
        "recommendations": recommendations
    }