"""
sodai_pipeline.py

DAG de Airflow para el pipeline de MLOps de SodAI Drinks.

- 'extract_and_preprocess' carga datos base + concatena nuevos batches.
- 'check_for_drift' implementa lógica real de K-S Test.
- 'generate_predictions' guarda los resultados en formato .csv.
"""

import sys
import os
from datetime import datetime, timedelta

# Importaciones LIGERAS del nivel superior
from airflow.decorators import dag, task
from airflow.utils.trigger_rule import TriggerRule

# --- 1. CONFIGURACIÓN DEL ENTORNO ---
AIRFLOW_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- 2. VARIABLES GLOBALES Y CONFIGURACIÓN ---
PROJECT_ROOT = os.path.abspath(os.path.join(AIRFLOW_PROJECT_PATH, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "data") 

# Archivos base (Entrega 1)
TRANS_PATH = os.path.join(DATA_DIR, "transacciones.parquet")
CUST_PATH = os.path.join(DATA_DIR, "clientes.parquet")
PROD_PATH = os.path.join(DATA_DIR, "productos.parquet")

# --- Rutas para Entrega 3 ---
BATCH_DIR_PATH = os.path.join(DATA_DIR, "batches")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MERGED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "df_merged_processed.parquet")
REFERENCE_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "df_merged_reference.parquet")
DRIFT_FEATURES_TO_MONITOR = ['items', 'size', 'num_deliver_per_week', 'num_visit_per_week']
# ----------------------------------------

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "sodai_drinks_production"
MODEL_REGISTRY_NAME = "sodai_model"

PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
PREDICTIONS_OUTPUT_PATH = os.path.join(PREDICTIONS_DIR, "latest_predictions.csv")

default_args = {
    'owner': 'DeepDrinkers',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 11), 
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0, 
    'retry_delay': timedelta(minutes=1),
}

# --- 3. DEFINICIÓN DEL DAG ---
@dag(
    dag_id="sodai_production_pipeline",
    default_args=default_args,
    description="Pipeline Condicional (Drift Detection) para SodAI Drinks",
    schedule=None,
    catchup=False,
    tags=['mlops', 'sodai', 'entrega2'],
)
def sodai_pipeline():
    """
    ### Pipeline de Producción SodAI Drinks (Arquitectura Condicional)
    
    Flujo:
    1. **extract_and_preprocess**: Carga datos base + concatena nuevos batches.
    2. **check_for_drift**: Ejecuta un K-S Test.
    3. **retrain_model**: Optimización y entrenamiento.
    4. **generate_predictions**: Genera predicciones y guarda en .csv.
    """

    def setup_task_path():
        """Función auxiliar para añadir la ruta del proyecto a sys.path dentro de la tarea."""
        sys.path.insert(0, AIRFLOW_PROJECT_PATH)

    @task
    def extract_and_preprocess() -> str:
        """
        Carga, concatena batches, limpia y fusiona los datos.
        Guarda el resultado en un Parquet y retorna la ruta.
        Crea el archivo de referencia si no existe.
        """
        import pandas as pd
        setup_task_path()
        try:
            from scripts import preprocessing
        except ImportError as e:
            print(f"Error importando 'scripts.preprocessing'. sys.path: {sys.path}")
            raise e

        print("--- Iniciando Tarea: extract_and_preprocess ---")
        
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) 
        os.makedirs(BATCH_DIR_PATH, exist_ok=True) 
        
        if not all(os.path.exists(p) for p in [TRANS_PATH, CUST_PATH, PROD_PATH]):
            raise FileNotFoundError("¡No se encontraron todos los archivos de datos base (Entrega 1)!")

        df_merged = preprocessing.load_and_preprocess_data(
            trans_path=TRANS_PATH,
            cust_path=CUST_PATH,
            prod_path=PROD_PATH,
            batch_dir_path=BATCH_DIR_PATH 
        )
        
        print(f"Guardando df_merged procesado en: {MERGED_DATA_PATH}")
        df_merged.to_parquet(MERGED_DATA_PATH, index=False)
        
        if not os.path.exists(REFERENCE_DATA_PATH):
            print(f"No se encontró archivo de referencia. Creando uno nuevo en: {REFERENCE_DATA_PATH}")
            df_reference = preprocessing.load_and_preprocess_data(
                trans_path=TRANS_PATH,
                cust_path=CUST_PATH,
                prod_path=PROD_PATH,
                batch_dir_path="" 
            )
            df_reference.to_parquet(REFERENCE_DATA_PATH, index=False)
        else:
            print(f"Archivo de referencia ya existe en: {REFERENCE_DATA_PATH}")
        
        print("--- Tarea Finalizada: extract_and_preprocess ---")
        return MERGED_DATA_PATH 

    @task.branch(task_id="check_for_drift")
    def check_for_drift(df_merged_path: str) -> str:
        """
        Ejecuta el K-S Test para detectar Data Drift.
        Retorna el ID de la siguiente tarea a ejecutar.
        """
        setup_task_path()
        try:
            from scripts import validation_drift
        except ImportError as e:
            print(f"Error importando 'scripts.validation_drift'. sys.path: {sys.path}")
            raise e
            
        print("--- Iniciando Detección de Drift ---")
        
        #drift_detected = True
        drift_detected = validation_drift.check_data_drift(
            new_data_path=df_merged_path,
            reference_data_path=REFERENCE_DATA_PATH,
            features_to_monitor=DRIFT_FEATURES_TO_MONITOR,
            p_value_threshold=0.05
        )
        
        if drift_detected:
            print("Resultado de la Rama: Drift detectado. Ejecutando 'retrain_model'.")
            return "retrain_model"
        else:
            print("Resultado de la Rama: Datos estables. Saltando a 'generate_predictions'.")
            return "generate_predictions"

    @task
    def retrain_model(df_merged_path: str) -> str:
        """
        Optimiza y entrena un nuevo modelo.
        """
        import pandas as pd
        import random 
        import gc 
        setup_task_path()
        try:
            from scripts import feature_engineering
            from scripts import model_training
        except ImportError as e:
            print(f"Error importando scripts: {e}")
            raise e

        print("--- Iniciando Tarea: retrain_model ---")
        
        print(f"Cargando datos procesados desde: {df_merged_path}")
        df_merged = pd.read_parquet(df_merged_path)
        
        all_weeks = sorted(df_merged['year_week'].unique())
        train_weeks = all_weeks[4:] 

        n_weeks_for_sample = 2
        if len(train_weeks) > n_weeks_for_sample:
            print(f"Tomando una muestra aleatoria de {n_weeks_for_sample} semanas.")
            random.seed(42)
            sampled_train_weeks = random.sample(train_weeks, n_weeks_for_sample)
        else:
            sampled_train_weeks = train_weeks
            
        print(f"Generando features para: {sampled_train_weeks}")
        df_features = feature_engineering.create_feature_matrix(
            df_merged=df_merged,
            target_weeks=sampled_train_weeks 
        )

        print("Liberando memoria...")
        del df_merged
        gc.collect() 

        n_trials_balanceados = 3
        print(f"Iniciando Optuna con {n_trials_balanceados} trials...")
        
        model_training.train_model( 
            df_features=df_features, 
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            model_registry_name=MODEL_REGISTRY_NAME,
            n_trials=n_trials_balanceados
        )
        
        print("--- Tarea Finalizada: retrain_model ---")
        return f"Modelo registrado como '{MODEL_REGISTRY_NAME}'"

    @task(trigger_rule=TriggerRule.NONE_FAILED)
    def generate_predictions(df_merged_path: str):
        """
        Genera predicciones y guarda en .csv para CodaLab.
        """
        import pandas as pd
        setup_task_path()
        try:
            from scripts import model_prediction
        except ImportError as e:
            print(f"Error importando scripts: {e}")
            raise e

        print("--- Iniciando Tarea: generate_predictions ---")
        
        print(f"Cargando datos desde: {df_merged_path}")
        df_merged = pd.read_parquet(df_merged_path)
        
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/latest"
        print(f"Usando modelo: {model_uri}")
        
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        
        # Llama a la función de predicción 
        model_prediction.generate_predictions(
            historical_data=df_merged,
            model_uri=model_uri,
            output_path=PREDICTIONS_OUTPUT_PATH, # La ruta ya es .csv
            mlflow_tracking_uri=MLFLOW_TRACKING_URI 
        )
        print("--- Tarea Finalizada: generate_predictions ---")
        return PREDICTIONS_OUTPUT_PATH

    # --- 4. DEFINICIÓN DEL FLUJO DE TAREAS (GRAFO) ---
    
    merged_data_filepath = extract_and_preprocess()
    drift_decision = check_for_drift(merged_data_filepath)
    retrain_op = retrain_model(merged_data_filepath)
    predict_op = generate_predictions(merged_data_filepath)
    
    drift_decision >> [retrain_op, predict_op]
    retrain_op >> predict_op

# Invocar la factory del DAG
sodai_pipeline()