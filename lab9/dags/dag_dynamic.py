import pendulum
from datetime import datetime
from airflow.models import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# Importaciones de Modelos de Scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from hiring_dynamic_functions import (
        create_folders,
        load_and_merge,
        split_data,
        train_model,
        evaluate_models
    )
except ImportError:
    print("Error importando hiring_dynamic_functions.py. Asegúrese de que esté en la carpeta dags.")
    # Placeholders para que el DAG se parsee
    def placeholder(*args, **kwargs): pass
    create_folders, load_and_merge, split_data, train_model, evaluate_models = (placeholder,) * 5


#  1. Inicialización del DAG  
with DAG(
    dag_id='hiring_dynamic',
    start_date=pendulum.datetime(2024, 10, 1, tz="UTC"),
    # Ejecutar el día 5 de cada mes a las 15:00 UTC (formato cron)
    schedule_interval='0 15 5 * *', 
    catchup=True,  # Habilitar backfill
    tags=['mlops', 'hiring', 'lab9_dynamic', 'parallel'],
    doc_md="Pipeline dinámico que entrena modelos en paralelo."
) as dag:

    #  2. Tarea de Inicio  
    start_pipeline = EmptyOperator(
        task_id='start_pipeline'
    )

    #  3. Tarea de Creación de Carpetas  
    create_folders_task = PythonOperator(
        task_id='create_folders_task',
        python_callable=create_folders
    )

    #  4. Tarea de Branching (Lógica de Fechas)  
    
    def check_date_branch(**kwargs):
        """
        Decide qué ramas de descarga ejecutar basado en la fecha de ejecución.
        """
        execution_date_str = kwargs['ds'] # 'YYYY-MM-DD'
        execution_date = datetime.strptime(execution_date_str, '%Y-%m-%d').date()
        
        # Fecha límite es 1 de Noviembre 2024
        threshold_date = datetime(2024, 11, 1).date()
        
        if execution_date < threshold_date:
            # Ejecuciones de Octubre
            print(f"Fecha {execution_date}: Descargando solo data_1.")
            return 'download_data_1_task' # ID de la tarea a ejecutar
        else:
            # Ejecuciones de Noviembre 2024 en adelante
            print(f"Fecha {execution_date}: Descargando data_1 y data_2.")
            return ['download_data_1_task', 'download_data_2_task'] # IDs de ambas tareas

    branching_task = BranchPythonOperator(
        task_id='check_date_branch',
        python_callable=check_date_branch
    )

    #  Tareas de Descarga (controladas por el Branching)  
    
    DATA_1_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    DATA_2_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
    
    # Plantillas Jinja para obtener la ruta 'raw' desde XCom
    output_path_1 = "{{ ti.xcom_pull(task_ids='create_folders_task')['raw_path'] }}/data_1.csv"
    output_path_2 = "{{ ti.xcom_pull(task_ids='create_folders_task')['raw_path'] }}/data_2.csv"
    
    download_data_1_task = BashOperator(
        task_id='download_data_1_task',
        bash_command=f"curl -sSL -o {output_path_1} {DATA_1_URL}"
    )
    
    download_data_2_task = BashOperator(
        task_id='download_data_2_task',
        bash_command=f"curl -sSL -o {output_path_2} {DATA_2_URL}"
    )

    #  5. Tarea de Fusión de Datos  
    load_and_merge_task = PythonOperator(
        task_id='load_and_merge',
        python_callable=load_and_merge,
        # Trigger: Se ejecuta si AL MENOS UNA de sus tareas padres (download_1 o download_2) tuvo éxito.
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    #  6. Tarea de División de Datos  
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    #  7. Tareas de Entrenamiento en Paralelo  
    
    # Lista para agrupar las tareas de entrenamiento
    training_tasks = []

    # Modelo 1: Random Forest
    train_rf_task = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_model,
        op_kwargs={
            'model': RandomForestClassifier(random_state=42),
            'model_name': 'random_forest'
        }
    )
    training_tasks.append(train_rf_task)

    # Modelo 2: Regresión Logística 
    train_lr_task = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_model,
        op_kwargs={
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'model_name': 'logistic_regression'
        }
    )
    training_tasks.append(train_lr_task)

    # Modelo 3: Gradient Boosting 
    train_gb_task = PythonOperator(
        task_id='train_gradient_boosting',
        python_callable=train_model,
        op_kwargs={
            'model': GradientBoostingClassifier(random_state=42),
            'model_name': 'gradient_boosting'
        }
    )
    training_tasks.append(train_gb_task)

    #  8. Tarea de Evaluación y Selección  
    evaluate_models_task = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )


    #  Definición de Dependencias 
    
    # 1. Inicio
    start_pipeline >> create_folders_task
    
    # 2. Branching para descarga
    create_folders_task >> branching_task >> [download_data_1_task, download_data_2_task]
    
    # 3. Merge 
    [download_data_1_task, download_data_2_task] >> load_and_merge_task
    
    # 4. Split
    load_and_merge_task >> split_data_task
    
    # 5. Entrenamiento en paralelo
    split_data_task >> training_tasks
    
    # 6. Evaluación 
    training_tasks >> evaluate_models_task