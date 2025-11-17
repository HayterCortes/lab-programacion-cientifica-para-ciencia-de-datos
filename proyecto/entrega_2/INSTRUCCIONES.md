# Instrucciones de Ejecución - Proyecto SodAI Drinks (Entrega 2)

Este documento detalla los pasos para configurar y ejecutar el proyecto completo, incluyendo el pipeline de MLOps (Airflow) y todas las aplicaciones web (FastAPI/Gradio/Docker).

## Entorno WSL2 

Este proyecto ha sido desarrollado y optimizado para un entorno **WSL2 (Ubuntu)**, se copia y se ejecuta desde el **sistema de archivos nativo de Linux** `~/projects/entrega_2/` o `/home/<su-usuario>/entrega_2/`.

---

## 1. Configuración del Entorno (WSL)

Siga estos pasos en una terminal de WSL (Ubuntu).

### 1.1. Mover el Proyecto 

Si la carpeta está en `/mnt/c/`, muévala a su `home` de Linux:

```bash
# Ejemplo:
mkdir ~/projects
mv /mnt/c/ruta/a/entrega_2 ~/projects/
cd ~/projects/entrega_2
````

### 1.2. Instalar Dependencias del Sistema

Se requiere **Python 3.11** (vía `deadsnakes`) y las herramientas de compilación necesarias.

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y build-essential python3.11 python3.11-venv python3.11-dev
```

### 1.3. Crear y Activar el Entorno Virtual

Se recomienda crear el `venv` *fuera* de la carpeta del proyecto para mantenerlo limpio.

```bash
# Desde su carpeta /home/<su-usuario>/
python3.11 -m venv airflow_env

# Activar el entorno
source ~/airflow_env/bin/activate
```

### 1.4. Instalar Dependencias de Python

Con el `venv` activado, navegue a la carpeta del proyecto e instale todo:

```bash
# cd ~/projects/entrega_2
pip install --upgrade pip

# 1. Instalar Airflow (con dependencias de webserver)
pip install --no-cache-dir "apache-airflow[webserver]"

# 2. Instalar el resto de la stack de MLOps y Apps
pip install mlflow docker fastapi "uvicorn[standard]" gradio requests \
            pandas pyarrow scikit-learn lightgbm optuna shap matplotlib \
            scipy tabulate langchain langchain-google-genai \
            langchain-experimental langchain-community google-generativeai
```

### 1.5. Configurar API Key del Bonus LLM 

El Bonus del Chatbot (`bonus/llm/`) requiere una API Key de Google Gemini.

1.  Cree el archivo `.env`:
    ```bash
    touch ~/projects/entrega_2/bonus/llm/.env
    ```
2.  Edite el archivo (ej. `code ~/projects/entrega_2/bonus/llm/.env`) y añada su clave:
    ```env
    GOOGLE_API_KEY=SU_API_KEY_DE_GOOGLE_AQUI
    ```

-----

## 2\. Ejecución de la Fase 1: Pipeline MLOps (Airflow + MLflow)

Este pipeline es condicional (detecta drift) y requiere un arranque para crear el primer modelo.

### 2.1. Inicializar y Configurar Airflow

1.  **Activar Entorno (si no está activo):**

    ```bash
    source ~/airflow_env/bin/activate
    ```

2.  **Inicializar Base de Datos de Airflow:**

    ```bash
    airflow db migrate
    ```

3.  **Configurar Airflow:**
    Abra el archivo de configuración de Airflow:

    ```bash
    code ~/airflow/airflow.cfg
    ```

    Realice los siguientes **dos cambios** y guarde el archivo:

      * **Desactivar DAGs de Ejemplo (Evita OOM):**
        ```ini
        # Buscar esta línea
        load_examples = True
        # Cambiar a:
        load_examples = False
        ```
      * **Apuntar a la Carpeta de DAGs del Proyecto:**
        ```ini
        # Buscar esta línea
        dags_folder = /home/<su-usuario>/airflow/dags
        # Cambiar a la ruta de su proyecto (reemplace <su-usuario>):
        dags_folder = /home/<su-usuario>/projects/entrega_2/airflow/dags
        ```

### 2.2. Lanzar Servicios (Airflow y MLflow)

Se requieren dos terminales (ambas con `source ~/airflow_env/bin/activate`).

  * **Terminal 1: MLflow Server (en segundo plano)**

    ```bash
    cd ~ 
    nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 127.0.0.1 --port 5000 > mlflow.log 2>&1 &
    ```

    *Acceso UI: `http://localhost:5000`*

  * **Terminal 2: Airflow Standalone**

    ```bash
    cd ~/projects/entrega_2
    airflow standalone
    ```

    *Acceso UI: `http://localhost:8080`*

### 2.3. Ejecución del Pipeline (Arranque en Frío)

El pipeline está diseñado para omitir el re-entrenamiento si no hay un modelo previo. Debemos forzar la creación de la **Versión 1** del modelo.

1.  **Paso 1: Forzar el Re-entrenamiento (Solo la primera vez)**

      * En VSCode, abra `~/projects/entrega_2/airflow/dags/sodai_pipeline.py`.

      * Vaya a la tarea `@task.branch(task_id="check_for_drift")` (aprox. línea 141).

      * **Comente** la llamada a la función `validation_drift` y **descomente** la línea `drift_detected = True`, como se muestra aquí:

        ```python
        # ... dentro de la tarea check_for_drift ...
        print("--- Iniciando Detección de Drift (MODO FORZADO PARA V1) ---")

        # --- ¡CAMBIO TEMPORAL PARA FORZAR EL ARRANQUE INICIAL! ---
        print("¡FORZANDO MODO DE RE-ENTRENAMIENTO PARA CREAR MODELO V1!")
        drift_detected = True 
        # drift_detected = validation_drift.check_data_drift(
        #     new_data_path=df_merged_path,
        #     reference_data_path=REFERENCE_DATA_PATH,
        #     features_to_monitor=DRIFT_FEATURES_TO_MONITOR,
        #     p_value_threshold=0.05
        # )
        # --- FIN DEL CAMBIO TEMPORAL ---

        if drift_detected:
            return "retrain_model"
        # ...
        ```

      * Guarde el archivo.

2.  **Paso 2: Ejecutar el Pipeline**

      * Vaya a la UI de Airflow (`localhost:8080`).
      * Active el DAG `sodai_production_pipeline` (interruptor "On").
      * Dispárelo (botón "Play" ▶️).
      * El pipeline se ejecutará, **forzará** la rama `retrain_model` y registrará el **Modelo Versión 1** en MLflow y generará `latest_predictions.csv`.

3.  **Paso 3: Volver al Modo Automático (¡IMPORTANTE\!)**

      * **Después** de que la ejecución termine (todo en verde), vuelva a `sodai_pipeline.py`.
      * **Deshaga el cambio** del Paso 1: comente `drift_detected = True` y descomente la llamada a `validation_drift.check_data_drift(...)`.
      * Guarde el archivo.

 Puede simular copiando archivos `batch_X.parquet` en la carpeta `data/batches/` y volviendo a disparar el DAG. El sistema detectará el drift (o no) automáticamente.

-----

## 3\. Ejecución de la Fase 2: Aplicaciones Docker

Asegúrese de que **Docker Desktop** esté corriendo e integrado con WSL.

### 3.1. Aplicación Principal (Predicciones de ML)

*Usa el `latest_predictions.csv` generado por el pipeline de Airflow.*

```bash
cd ~/projects/entrega_2/app
docker-compose up --build
```

  * **Acceso UI:** `http://localhost:7860`

### 3.2. Bonus: Sistema de Recomendación

*Usa los datos crudos (`transacciones.parquet`).*

```bash
cd ~/projects/entrega_2/bonus/recsys
docker-compose up --build
```

  * **Acceso UI:** `http://localhost:7861`

### 3.3. Bonus: Chatbot Conversacional (LLM)

*Usa los datos crudos y requiere la API Key de Google (ver 1.5).*

```bash
cd ~/projects/entrega_2/bonus/llm
docker-compose up --build
```

  * **Acceso UI:** `http://localhost:7862`

<!-- end list -->
