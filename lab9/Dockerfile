# imagen base de Python 3.10-slim
FROM python:3.10-slim

# 1. Configurar variables de entorno de Airflow
ENV AIRFLOW_HOME=/opt/airflow
# Deshabilitar DAGs de ejemplo y pausado inicial
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False

# 2. Instalar dependencias del sistema (curl)
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Instalar Airflow y dependencias de Python
RUN pip install --no-cache-dir \
    apache-airflow==2.8.4 \
    psycopg2-binary \
    pandas==2.1.4 \
    scikit-learn==1.4.2 \
    joblib==1.4.2

# 4. Instalar Gradio 
RUN pip install --no-cache-dir --upgrade \
    gradio==4.19.2 \
    "email-validator>=2.0" \
    huggingface-hub==0.19.4

# 4. Copiar la carpeta dags (y su contenido) al contenedor
COPY dags ${AIRFLOW_HOME}/dags

# 5. Exponer el puerto de Airflow y el puerto de Gradio
EXPOSE 8080
EXPOSE 7860

# 6. Establecer el directorio de trabajo
WORKDIR ${AIRFLOW_HOME}

# 7. Configurar y usar 'airflow standalone' 
# Esto inicializa la DB, crea un usuario (admin/admin) y lanza scheduler + webserver
ENV AIRFLOW__WEBSERVER__SECRET_KEY="super-secret-key-for-lab"
ENV _AIRFLOW_DB_UPGRADE="true"
ENV _AIRFLOW_WWW_USER_CREATE="true"
ENV _AIRFLOW_WWW_USER_USERNAME="admin"
ENV _AIRFLOW_WWW_USER_PASSWORD="admin"
ENV _AIRFLOW_WWW_USER_ROLE="Admin"

ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor    
CMD ["airflow", "standalone"]