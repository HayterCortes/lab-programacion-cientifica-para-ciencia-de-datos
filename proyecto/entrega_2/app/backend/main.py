"""
Backend (FastAPI) para la aplicación SodAI Drinks.

"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os

# --- Configuración ---
PREDICTIONS_FILE_PATH = "/app/data/latest_predictions.csv"

# Variable global para mantener los datos en memoria
predictions_data = {
    "df": None
}

# --- Funciones de Carga de Datos ---

def load_predictions() -> pd.DataFrame:
    """
    Carga el archivo CSV de predicciones en un DataFrame de Pandas.
    """
    if not os.path.exists(PREDICTIONS_FILE_PATH):
        print(f"ADVERTENCIA: No se encontró el archivo de predicciones en {PREDICTIONS_FILE_PATH}.")
        print("Sirviendo un DataFrame vacío. Por favor, ejecute el pipeline de Airflow.")
        return pd.DataFrame(columns=["customer_id", "product_id", "probability"])
        
    print(f"Cargando predicciones desde {PREDICTIONS_FILE_PATH}...")
    
    df = pd.read_csv(PREDICTIONS_FILE_PATH)
    
    # Asegurarnos de que las columnas esperadas existan
    if not all(col in df.columns for col in ["customer_id", "product_id", "probability"]):
        raise HTTPException(status_code=500, detail="El archivo CSV no tiene las columnas esperadas.")
        
    print(f"Predicciones cargadas. Forma: {df.shape}")
    return df

# --- Eventos de Ciclo de Vida de la App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manejador de eventos de ciclo de vida.
    Se ejecuta al iniciar la aplicación para cargar el modelo/datos.
    """
    print("Iniciando aplicación...")
    predictions_data["df"] = load_predictions()
    yield
    # Código de limpieza (si fuera necesario)
    print("Apagando aplicación...")
    predictions_data["df"] = None

# --- Creación de la App ---

app = FastAPI(
    title="SodAI Drinks - API de Recomendaciones",
    description="Sirve las 5 predicciones de productos con mayor probabilidad para un cliente dado.",
    version="1.0.0",
    lifespan=lifespan # Carga los datos al iniciar
)

# --- Endpoints de la API ---

@app.get("/")
def read_root():
    """Endpoint de salud para verificar que la API esté viva."""
    return {"status": "SodAI Drinks API está viva y funcionando."}

@app.get("/recommendations/{customer_id}")
def get_recommendations(customer_id: int):
    """
    Obtiene las 5 recomendaciones de productos con mayor probabilidad
    para un ID de cliente específico.
    """
    df = predictions_data["df"]
    
    if df is None or df.empty:
        raise HTTPException(status_code=503, 
                            detail="Los datos de predicciones no están cargados. Ejecute el pipeline de Airflow.")
    
    # Filtrar por cliente
    client_preds = df[df["customer_id"] == customer_id].copy()
    
    if client_preds.empty:
        raise HTTPException(status_code=404, 
                            detail=f"No se encontraron predicciones para el customer_id: {customer_id}")
    
    # Ordenar por probabilidad y tomar el top 5
    top_5_preds = client_preds.sort_values(by="probability", ascending=False).head(5)
    
    # Convertir a un formato JSON 
    recommendations = top_5_preds[['product_id', 'probability']].to_dict('records')
    
    return {
        "customer_id": customer_id,
        "recommendations": recommendations
    }