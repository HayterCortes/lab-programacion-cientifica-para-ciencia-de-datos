"""
model_training.py

Este script contiene toda la lógica para el entrenamiento y optimización
del modelo LightGBM del proyecto SodAI Drinks, refactorizado de la Entrega 1.

"""

import pandas as pd
import numpy as np
import optuna
import lightgbm as lgbm
import mlflow
import mlflow.sklearn
import warnings
import os 
import gc 
from typing import Dict, Any, Tuple

import matplotlib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ignorar warnings de Optuna y Sklearn para un log más limpio
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. IMPORTAR CONFIGURACIÓN ---
try:
    from .config import (
        NUMERIC_FEATURES, 
        CATEGORICAL_FEATURES, 
        ID_COLS, 
        TARGET_COL
    )
except ImportError:
    # Fallback para si el script se ejecuta directamente
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


# --- 2. FUNCIÓN DE CONSTRUCCIÓN DEL PREPROCESADOR ---

def get_preprocessor_pipeline(num_imputer_strategy: str, scaler: str) -> ColumnTransformer:
    """
    Construye y devuelve el ColumnTransformer para el preprocesamiento de features.
    """
    
    if scaler == 'StandardScaler':
        scaler_obj = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler_obj = MinMaxScaler()
    elif scaler == 'RobustScaler':
        scaler_obj = RobustScaler()
    elif scaler == 'None':
        scaler_obj = 'passthrough'
    else:
        raise ValueError(f"Escalador '{scaler}' no reconocido.")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
        ('scaler', scaler_obj)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor


# --- 3. FUNCIÓN OBJETIVO DE OPTUNA ---

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Función objetivo para la optimización de Optuna, diseñada para maximizar el F1-Score.
    """
    
    num_imputer_strategy = trial.suggest_categorical('num_imputer_strategy', ['mean', 'median'])
    scaler_choice = trial.suggest_categorical('scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None'])

    lgbm_params = {
        'objective': 'binary',
        'metric': 'f1',
        'random_state': 42,
        'n_jobs': 1,  
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80), 
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False])
    }
    
    preprocessor = get_preprocessor_pipeline(num_imputer_strategy, scaler_choice)
    model = lgbm.LGBMClassifier(**lgbm_params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(pipeline, X, y, scoring='f1', cv=cv, n_jobs=1)
        f1_mean = np.mean(scores)
        return f1_mean
    except Exception as e:
        print(f"Trial fallido con error: {e}. Retornando 0.0")
        return 0.0


# --- 4. FUNCIÓN MAESTRA DE ENTRENAMIENTO (CON OPTUNA) ---

def train_model(
    df_features: pd.DataFrame, 
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    model_registry_name: str,
    n_trials: int = 50
) -> Pipeline:
    """
    Orquesta el proceso completo de entrenamiento CON OPTUNA.
    """
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap

    print("--- Iniciando Proceso de Entrenamiento y Optimización ---")

    X_train = df_features.drop(columns=[TARGET_COL] + ID_COLS)
    y_train = df_features[TARGET_COL]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name="Balanced Training Run (with Optuna)") as run:
        run_id = run.info.run_id
        print(f"MLflow Run iniciado con ID: {run_id}")
        mlflow.log_param("data_shape", X_train.shape)
        mlflow.log_param("n_trials", n_trials)

        print(f"Iniciando optimización con {n_trials} trials para maximizar el F1-Score...")
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, X_train, y_train), 
            n_trials=n_trials
        )
        
        print("Optimización completada.")

        best_params = study.best_trial.params
        best_f1_cv = study.best_trial.value
        
        print(f"Mejor F1-Score (CV): {best_f1_cv:.6f}")
        print(f"Mejores Hiperparámetros: {best_params}")

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_cv", best_f1_cv)
        
        print("Entrenando modelo final con los mejores hiperparámetros...")
        
        final_num_imputer = best_params.pop('num_imputer_strategy')
        final_scaler = best_params.pop('scaler')
        
        final_lgbm_params = {
            'objective': 'binary',
            'metric': 'f1',
            'random_state': 42,
            'n_jobs': 1, 
            **best_params 
        }
        
        final_preprocessor = get_preprocessor_pipeline(final_num_imputer, final_scaler)
        final_model = lgbm.LGBMClassifier(**final_lgbm_params)
        
        final_pipeline = Pipeline(steps=[
            ('preprocessor', final_preprocessor),
            ('classifier', final_model)
        ])
        
        final_pipeline.fit(X_train, y_train)
        print("Entrenamiento final completado.")

        print("Iniciando generación de gráficos de interpretabilidad (SHAP)...")
        try:
            model = final_pipeline.named_steps['classifier']
            preprocessor = final_pipeline.named_steps['preprocessor']
            
            n_shap_samples = min(2000, len(X_train))
            shap_sample_df = X_train.sample(n=n_shap_samples, random_state=42)
            
            shap_sample_processed = preprocessor.transform(shap_sample_df)
            feature_names = preprocessor.get_feature_names_out()
            shap_sample_processed_df = pd.DataFrame(shap_sample_processed, columns=feature_names)

            explainer = shap.TreeExplainer(model)
            shap_values_obj = explainer(shap_sample_processed_df)
            
            shap_values_to_plot = shap_values_obj.values
            
            if shap_values_obj.values.ndim == 3:
                # Caso: Clasificación binaria con output para ambas clases
                print("SHAP: Detectado output de clasificación binaria (3D). Usando valores de la Clase 1.")
                shap_values_to_plot = shap_values_obj.values[..., 1]
            else:
                # Caso: Regresión o clasificación binaria con un solo output
                print("SHAP: Detectado output de puntuación única (2D). Usando valores directamente.")

            shap_summary_path = "shap_summary_plot.png"
            shap.summary_plot(shap_values_to_plot, shap_sample_processed_df, show=False)
            plt.savefig(shap_summary_path, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(shap_summary_path, artifact_path="model_interpretability")
            os.remove(shap_summary_path) 
            print("Gráfico de resumen de SHAP guardado y registrado en MLflow.")

        except Exception as e:
            print(f"Advertencia: No se pudo generar el gráfico de SHAP. Error: {e}")
        
        print("Loggeando pipeline final en MLflow...")
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=model_registry_name
        )
        print(f"Pipeline loggeado y registrado como '{model_registry_name}'")
        
    print("--- Proceso de Entrenamiento y Optimización Finalizado ---")
    return final_pipeline
