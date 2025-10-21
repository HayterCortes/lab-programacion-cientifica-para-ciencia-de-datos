import uvicorn
import pickle
import pandas as pd
import mlflow
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Definición del Modelo Pydantic (Entrada de Datos)   

# Define la estructura de entrada de la solicitud POST
class WaterFeatures(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

    # Ejemplo para la documentación de la API
    class Config:
        schema_extra = {
            "example": {
                "ph": 10.316,
                "Hardness": 217.266,
                "Solids": 10676.508,
                "Chloramines": 3.445,
                "Sulfate": 397.754,
                "Conductivity": 492.206,
                "Organic_carbon": 12.812,
                "Trihalomethanes": 72.281,
                "Turbidity": 3.407
            }
        }

# 2. Carga de Artefactos (Modelo y Scaler)   

# Constantes
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "water_potability.csv")
EXPERIMENT_NAME = "Optimización Potabilidad Agua XGBoost"
FEATURE_NAMES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

def get_fitted_scaler():
    
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} no encontrado.")
        return None

    # Replicar pasos de preprocesamiento de optimize.py
    df_imputed = df.fillna(df.median())
    y = df_imputed['Potability']
    X = df_imputed.drop('Potability', axis=1)
    
    # Replicar el mismo split para obtener el mismo set de entrenamiento
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    scaler.fit(X_train[FEATURE_NAMES]) 
    print("Scaler re-entrenado y listo.")
    return scaler

def get_model():
    """
    Busca en MLflow el "Run de reporte final" y carga el modelo .pkl 
    que fue serializado y guardado en sus artefactos.
    """
    print("Buscando el mejor modelo en MLflow...")
    
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise Exception(f"Experimento '{EXPERIMENT_NAME}' no encontrado. Ejecute optimize.py primero.")
            
        # Buscar el run específico que guarda el reporte final
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'Mejor_Modelo_y_Reporte_Final'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs_df.empty:
            raise Exception("No se encontró el 'Run de reporte final'.")
        
        best_run_id = runs_df.iloc[0]["run_id"]
        
        # Ruta al artefacto .pkl específico guardado en optimize.py
        model_uri = f"runs:/{best_run_id}/models/best_potability_model.pkl"
        
        print(f"Cargando modelo desde {model_uri}")
        
        # Descargar el artefacto (pickle) a una carpeta temporal
        local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        
        # Cargar el modelo desde el archivo pickle descargado
        with open(local_model_path, "rb") as f:
            model = pickle.load(f)
        
        print("Modelo cargado exitosamente.")
        return model
        
    except Exception as e:
        print(f"Error fatal al cargar el modelo desde MLflow: {e}")
        print("Asegúrese de estar en el directorio 'lab8' (que contiene 'mlruns')")
        return None

# Cargar el modelo y el scaler UNA VEZ al iniciar la API
print("Iniciando carga de artefactos de modelo...")
model = get_model()
scaler = get_fitted_scaler()

# 3. Creación de la App FastAPI y Endpoints   

app = FastAPI(
    title="API de Potabilidad del Agua 💧",
    description="API para predecir si una muestra de agua es potable usando un modelo XGBoost optimizado con Optuna y MLflow.",
    version="1.0.0"
)

@app.get("/", summary="Descripción de la API y Modelo")
def get_home():
    """
    Endpoint principal que describe la API.
    """
    return {
        "proyecto": "API de Predicción de Potabilidad del Agua",
        "descripcion_problema": "Esta API utiliza un modelo de Machine Learning (XGBoost) para predecir si una muestra de agua es potable (apta para consumo humano) o no, basándose en 9 mediciones químicas.",
        "modelo": "XGBoost Classifier optimizado con Optuna y loggeado con MLflow.",
        "input (POST /potabilidad/)": "Un objeto JSON con 9 atributos: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity.",
        "output (POST /potabilidad/)": "Un objeto JSON con la predicción: {'potabilidad': 1} (potable) o {'potabilidad': 0} (no potable).",
        "documentacion_interactiva": "Ir a /docs"
    }

@app.post("/potabilidad/", summary="Predecir la potabilidad de una muestra de agua")
def predict_potability(features: WaterFeatures):
    """
    Recibe una muestra de agua (9 atributos) y devuelve
    la predicción de potabilidad.
    """
    
    if model is None or scaler is None:
        # Manejar fallo en la carga inicial
        return {"error": "El modelo y/o el scaler no pudieron cargarse."}, 503

    try:
        # 1. Convertir Pydantic model a DataFrame
        
        input_data = {name: getattr(features, name) for name in FEATURE_NAMES}
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # 2. Escalar los datos (usando el scaler re-entrenado)
        input_scaled = scaler.transform(input_df)
        
        # 3. Realizar la predicción
        prediction_array = model.predict(input_scaled)
        
        # 4. Formatear la salida
        prediction = int(prediction_array[0])
        return {"potabilidad": prediction}

    except Exception as e:
        # Capturar error durante la predicción
        return {"error": f"Error durante la predicción: {str(e)}"}, 500

# 4. Punto de Entrada para Ejecutar la API   

if __name__ == "__main__":
    """
    Permite ejecutar la API directamente con `python main.py`
    """
    print("Iniciando servidor FastAPI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)