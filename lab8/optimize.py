import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import pickle
import json
import subprocess
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "water_potability.csv")

# Nombre del experimento personalizado en MLflow
EXPERIMENT_NAME = "Optimización Potabilidad Agua XGBoost" 

def load_preprocess_data(filepath: str):
    """
    Carga y preprocesa los datos.
    
    """
    print(f"Cargando datos desde {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: El archivo '{filepath}' no se encuentra.")
        return None, None, None, None, None
    
    # 1. Imputación de valores nulos:
    # fillna con la mediana de cada columna.
    # df.median() calcula la mediana de forma optimizada.
    print("Imputando valores nulos con la mediana...")
    df_imputed = df.fillna(df.median())
    
    # 2. Separar features (X) y target (y)
    y = df_imputed['Potability'].astype(int)
    X = df_imputed.drop('Potability', axis=1)
    
    # nombres de las columnas para el gráfico de importancia
    feature_names = X.columns.to_list()

    # 3. Separar en Train/Test
    # stratify=y para mantener la proporción de clases en ambos sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Escalar los datos
    print("Escalando datos (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de nuevo a DataFrame para mantener nombres de columnas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    print("Preprocesamiento completado.")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def objective(trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Función objetivo que Optuna intentará maximizar.
    Cada ejecución (trial) se registra como un 'run' anidado en MLflow.
    """
    
    # 1. Definir el espacio de búsqueda de hiperparámetros
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss', # Métrica de evaluación interna de XGBoost
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
        
    }
    
    # 2. Crear un nombre de 'run' interpretable
    run_name = f"XGB_Trial_{trial.number}_LR_{params['learning_rate']:.3f}_Depth_{params['max_depth']}"
    
    # 3. Iniciar el 'run' en MLflow
    # 'nested=True' para que estos runs se agrupen bajo el 'run' padre
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        
        # Log manual de parámetros 
        mlflow.log_params(params)
        
        model = xgb.XGBClassifier(**params)
        
        # 4. Entrenar y evaluar usando Cross-Validation (para Optuna)
        
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        f1_score_mean = scores.mean()
        
        # 5. Registrar la métrica objetivo (f1-score) como "valid_f1"
        mlflow.log_metric("valid_f1", f1_score_mean)

        model.fit(X_train, y_train)
        
    return f1_score_mean

def optimize_model():
    """
    Función principal que orquesta todo el proceso:
    1. Carga datos
    2. Configura MLflow
    3. Ejecuta el estudio de Optuna
    4. Encuentra el mejor modelo
    5. Guarda todos los artefactos solicitados en un 'run' final.
    """
    
    warnings.filterwarnings("ignore")
    
    # 1. Cargar y preprocesar datos
    X_train, _, y_train, _, feature_names = load_preprocess_data(DATA_FILE)
    if X_train is None:
        return None # Detener ejecución si el archivo no se encontró

    # 2. Configurar MLflow
    # Crea o selecciona el experimento con el nombre personalizado
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # autologging de XGBoost.
    # registrará métricas, parámetros y el modelo (artefacto)
    # en cada 'run' iniciado con 'mlflow.start_run'.
    mlflow.xgboost.autolog() 
    
    # 3. Ejecutar el estudio de Optuna
    print("Iniciando optimización de hiperparámetros con Optuna...")
    study = optuna.create_study(direction="maximize") # maximizar F1-Score
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=50, # Número de iteraciones 
        show_progress_bar=True
    )
    
    print(f"Optimización completada. Mejor F1-score (CV): {study.best_value:.4f}")
    
    # 4. Obtener el mejor 'run' de MLflow
    print("Buscando el mejor 'run' en MLflow...")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["metrics.valid_f1 DESC"],
        max_results=1
    )
    
    best_run_id = runs_df.iloc[0]["run_id"]
    print(f"Mejor 'run' encontrado: {best_run_id}")
    
    # 5. Cargar el mejor modelo 
    best_model_uri = f"runs:/{best_run_id}/model"
    best_model = mlflow.xgboost.load_model(best_model_uri)
    
    # 6. Guardar artefactos finales en un nuevo 'run' de resumen
    print("Guardando artefactos finales (plots, modelo .pkl, config, requirements)...")
    
    with mlflow.start_run(run_name="Mejor_Modelo_y_Reporte_Final") as final_run:
        
        # Loggear métricas y parámetros del mejor modelo 
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_valid_f1", study.best_value)
        mlflow.set_tag("best_run_id", best_run_id) # Referencia al run original
        
        # Paths de artefactos 
        plots_path = "plots"
        models_path = "models"
        
        # Guardar y loggear gráficos de Optuna en /plots 
        try:
            # Gráfico 1: Historial de optimización
            fig_hist = plot_optimization_history(study)
            fig_hist_path = "optuna_history.png"
            fig_hist.write_image(fig_hist_path) # Guardar como imagen
            mlflow.log_artifact(fig_hist_path, artifact_path=plots_path)

            # Gráfico 2: Importancia de hiperparámetros
            fig_param_imp = plot_param_importances(study)
            fig_param_imp_path = "optuna_param_importances.png"
            fig_param_imp.write_image(fig_param_imp_path)
            mlflow.log_artifact(fig_param_imp_path, artifact_path=plots_path)
            
            # Gráfico 3: Gráfico de corte (slice)
            fig_slice = plot_slice(study)
            fig_slice_path = "optuna_slice.png"
            fig_slice.write_image(fig_slice_path)
            mlflow.log_artifact(fig_slice_path, artifact_path=plots_path)

        except ImportError:
            print("ImportError")
        except Exception as e:
            print(f"Error al guardar gráficos de Optuna: {e}")

        # Guardar y loggear Importancia de Variables (XGBoost) en /plots 
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(best_model, ax=ax, title="Importancia de Variables (XGBoost)")
        feature_imp_path = "feature_importance.png"
        plt.savefig(feature_imp_path, bbox_inches='tight')
        plt.close(fig) # Cerrar la figura para liberar memoria
        mlflow.log_artifact(feature_imp_path, artifact_path=plots_path)
        
        # Guardar y loggear Configuración del modelo (JSON) en /plots 
        config_path = "best_model_config.json"
        with open(config_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        mlflow.log_artifact(config_path, artifact_path=plots_path)
        
        # Serializar y loggear el mejor modelo en /models 
        model_path = "best_potability_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(model_path, artifact_path=models_path)
        
        # Guardar y loggear las versiones de las librerías 
        req_path = "requirements.txt"
        try:
            with open(req_path, "w") as f:
                subprocess.run(["pip", "freeze"], stdout=f, text=True, check=True)
            mlflow.log_artifact(req_path)
        except Exception as e:
            print(f"No se pudo guardar 'requirements.txt': {e}")
            
        # Limpieza de archivos locales temporales 
        local_files_to_clean = [
            "optuna_history.png", "optuna_param_importances.png", 
            "optuna_slice.png", "feature_importance.png",
            "best_model_config.json", "best_potability_model.pkl",
            "requirements.txt"
        ]
        for f in local_files_to_clean:
            if os.path.exists(f):
                os.remove(f)

    print(f"Experimento: {EXPERIMENT_NAME}")
    print(f"Mejor 'run' de trial: {best_run_id}")
    print(f"Run de reporte final: {final_run.info.run_id}")
    print(f"Modelo serializado guardado en 'models/best_potability_model.pkl' dentro del run final")
    
    return best_model

# Punto de entrada para ejecución del script 
if __name__ == "__main__":
    optimize_model()