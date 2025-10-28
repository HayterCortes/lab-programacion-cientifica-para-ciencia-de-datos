from datetime import datetime
from airflow.models import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

try:
    from hiring_functions import (
        create_folders,
        split_data,
        preprocess_and_train,
        gradio_interface
    )
except ImportError:
    print("Error importando hiring_functions.py. Asegúrese de que esté en la carpeta dags.")
    def placeholder(): pass
    create_folders, split_data, preprocess_and_train, gradio_interface = placeholder, placeholder, placeholder, placeholder

# 0. Inicialización del DAG 
with DAG(
    dag_id='hiring_lineal',
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,  # Ejecución manual
    catchup=False,           # Sin backfill
    tags=['mlops', 'hiring', 'lab9_lineal'],
    doc_md="Pipeline lineal para predicción de contratación (Lab 9)"
) as dag:

    # 1. Tarea de Inicio 
    start_pipeline = EmptyOperator(
        task_id='start_pipeline'
    )

    # 2. Tarea de Creación de Carpetas 
    # Esta tarea usa la función 'create_folders'
    # y pasa las rutas vía XCom automáticamente.
    create_folders_task = PythonOperator(
        task_id='create_folders', # Este ID es usado por las tareas siguientes
        python_callable=create_folders
    )

    # 3. Tarea de Descarga de Datos 
    DATA_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    
    output_path = "{{ ti.xcom_pull(task_ids='create_folders')['raw_path'] }}/data_1.csv"
    
    download_data_task = BashOperator(
        task_id='download_data',
        bash_command=f"curl -sSL -o {output_path} {DATA_URL}",
        doc_md="Descarga data_1.csv desde Gitlab."
    )

    # 4. Tarea de División de Datos 
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        doc_md="Aplica Hold-Out (80/20) estratificado."
    )

    # 5. Tarea de Entrenamiento 
    # Esta tarea guarda el modelo y pasa la ruta vía XCom.
    preprocess_and_train_task = PythonOperator(
        task_id='preprocess_and_train', # ID usado por la tarea de Gradio
        python_callable=preprocess_and_train,
        doc_md="Preprocesa datos, entrena RandomForest y guarda pipeline."
    )

    # 6. Tarea de Interfaz Gradio 
    gradio_interface_task = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface,
        doc_md="Lanza la interfaz de Gradio en un link público."
    )

    # Definición de Dependencias (Pipeline Lineal) 
    (
        start_pipeline 
        >> create_folders_task 
        >> download_data_task 
        >> split_data_task 
        >> preprocess_and_train_task 
        >> gradio_interface_task
    )