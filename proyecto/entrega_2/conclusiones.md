# Conclusiones de la Entrega 2: Implementación de MLOps

Esta sección presenta las reflexiones sobre los aprendizajes, desafíos y oportunidades identificados durante la aplicación de prácticas de MLOps al proyecto SodAI Drinks. El proceso implicó la transición de un modelo de Machine Learning a un pipeline de producción automatizado y eficiente.

---

## 1. Impacto de las Herramientas de Tracking y Despliegue

La adopción de herramientas específicas de MLOps (MLflow y Docker) transformó el proyecto de un notebook experimental a un producto de software.

* **Tracking con MLflow:** El uso de MLflow fue fundamental para la trazabilidad.
    * **Versionado de Modelos:** Al registrar el pipeline (preprocesador + modelo) en el "Model Registry", se desacopla el entrenamiento de la predicción. La tarea de inferencia siempre solicita la versión `latest`, sin importar cuándo se entrenó.
    * **Interpretabilidad Automatizada:** Se logró integrar la generación de gráficos SHAP dentro del pipeline, registrándolos como artefactos en cada ejecución. Esto permite auditar no solo el rendimiento (F1-Score), sino también el *comportamiento* del modelo a lo largo del tiempo.

* **Despliegue con Docker:** Docker solucionó los problemas de compatibilidad.
    * **Arquitectura de Microservicios:** Se separó la lógica en contenedores especializados: una API (`backend`) para el procesamiento de datos y una interfaz (`frontend`) para la interacción con el usuario.
    * **Orquestación:** `docker-compose` permite levantar aplicaciones (como el Sistema de Recomendación y el Chatbot), gestionando redes internas y volúmenes de datos automáticamente.

---

## 2. Desafíos Técnicos y Soluciones (Ingeniería de Datos)

El mayor desafío de esta entrega fue la **gestión de recursos** en un entorno limitado. Las tareas guardan los datos intermedios en disco (`.parquet`) y solo pasan la ruta del archivo entre ellas. Para la ingeniería de features, se reemplazaron operaciones costosas usadas en la Entrega 1 como `pd.merge` por `pd.Series.map`, lo que redujo tiempos de ejecución de minutos a segundos.

---

## 3. Aporte de Airflow

* **Arquitectura Condicional (Branching):**
    A diferencia de un script lineal, se implementó un **Grafo Dirigido con Ramificaciones**. El pipeline ahora incluye una tarea de decisión (`check_for_drift`) que evalúa estadísticamente los datos entrantes (Test K-S).
    * **Si hay Drift:** El flujo se dirige automáticamente a la tarea de re-entrenamiento y optimización.
    * **Si NO hay Drift:** El sistema salta el re-entrenamiento costoso y pasa directamente a predecir usando el modelo vigente.

* **Ingesta Incremental:**
    El pipeline fue diseñado para ser evolutivo. La tarea de extracción es capaz de detectar nuevos archivos ("batches") en el directorio de datos y concatenarlos automáticamente con la historia, permitiendo una simulación realista de un entorno de producción.

---

## 4. Lecciones Aprendidas y Próximos Pasos

La transición de ML a MLOps reveló que un modelo exitoso es una pequeña parte de un sistema funcional.

* **Estado Final:**
    El proyecto se entrega con una arquitectura "Future-Proof". El pipeline ya está configurado para manejar la llegada de nuevos datos y tomar decisiones autónomas de re-entrenamiento, y la salida se ha estandarizado a `.csv` para integración con sistemas externos.