import pandas as pd
import joblib
import gradio as gr
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Creación de Carpetas 

def create_folders(**kwargs):
    """
    Crea una estructura de carpetas basada en la fecha de ejecución (ds).
    Pasa las rutas de las carpetas creadas vía XCom.
    """
    # fecha de ejecución del contexto de Airflow
    execution_date = kwargs['ds']

    # ruta base en el directorio de trabajo actual
    base_path = Path.cwd() / execution_date

    # subcarpetas
    raw_path = base_path / 'raw'
    splits_path = base_path / 'splits'
    models_path = base_path / 'models'

    # todas las carpetas
    for path in [base_path, raw_path, splits_path, models_path]:
        path.mkdir(parents=True, exist_ok=True)

    print(f"Carpetas creadas en: {base_path}")

    # diccionario con las rutas para que Airflow las pase por XCom
    return {
        'base_path': str(base_path),
        'raw_path': str(raw_path),
        'splits_path': str(splits_path),
        'models_path': str(models_path)
    }

# 2. División de Datos 

def split_data(**kwargs):
    """
    Lee los datos de 'raw', aplica hold-out (80/20) estratificado
    y guarda los sets de train/test en 'splits'.

    Depende de:
    - XCom de la tarea 'create_folders'.
    - El archivo 'data_1.csv' debe estar presente en la carpeta 'raw'
      correspondiente a la fecha de ejecución.
    """
    # rutas desde la tarea anterior vía XCom
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='create_folders') 

    raw_path = Path(paths['raw_path'])
    splits_path = Path(paths['splits_path'])

    input_file = raw_path / 'data_1.csv'

    # Verificación de que el archivo existe
    if not input_file.exists():
        error_msg = (
            f"ERROR: El archivo '{input_file}' no se encuentra. "
            "Por favor, asegúrese de que 'data_1.csv' esté en la carpeta 'raw' "
            "antes de ejecutar esta tarea."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)

    print(f"Leyendo datos de: {input_file}")
    df = pd.read_csv(input_file)

    # features (X) y target (y)
    TARGET = 'HiringDecision'
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # Split 80/20 estratificado con semilla
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42,  # Semilla fija
        stratify=y         # Mantener proporción de la variable objetivo
    )

    # X e y para guardar los archivos CSV completos
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # rutas de salida
    train_output = splits_path / 'train.csv'
    test_output = splits_path / 'test.csv'

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"Datos de entrenamiento guardados en: {train_output}")
    print(f"Datos de prueba guardados en: {test_output}")

# 3. Preprocesamiento y Entrenamiento 

def preprocess_and_train(**kwargs):
    """
    Lee los datos de 'splits', define un pipeline de preprocesamiento
    con ColumnTransformer, entrena un RandomForest y guarda el pipeline
    entrenado en 'models'. Finalmente, imprime métricas de evaluación.

    Depende de:
    - XCom de la tarea 'create_folders'.
    - Archivos 'train.csv' y 'test.csv' en la carpeta 'splits'.
    """
    # rutas desde la tarea 'create_folders'
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='create_folders') 

    splits_path = Path(paths['splits_path'])
    models_path = Path(paths['models_path'])

    # datos de entrenamiento y prueba
    print("Leyendo sets de entrenamiento y prueba...")
    train_df = pd.read_csv(splits_path / 'train.csv')
    test_df = pd.read_csv(splits_path / 'test.csv')

    # features y target
    TARGET = 'HiringDecision'
    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]
    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]

    # Definición del Pipeline de Preprocesamiento 
    # variables numéricas y categóricas

    numeric_features = [
        'Age', 'ExperienceYears', 'PreviousCompanies', 
        'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore'
    ]

    # estas son variables categóricas (aunque estén como númericas)
    categorical_features = ['Gender', 'EducationLevel', 'RecruitmentStrategy']

    # transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') # Ignora categorías no vistas en train

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Dejar pasar columnas no especificadas (aunque aquí usamos todas)
    )

    # Definición del Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42)) # Modelo con semilla
    ])

    # Entrenamiento 
    print("Iniciando entrenamiento del modelo...")
    pipeline.fit(X_train, y_train)
    print("Entrenamiento finalizado.")

    # Evaluación 
    print("Evaluando modelo en el conjunto de prueba...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # f1-score para la clase positiva (1 = Contratado)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score (Clase 1 - Contratado): {f1:.4f}")

    # Guardado del Modelo 
    model_path = models_path / 'hiring_model.joblib'
    joblib.dump(pipeline, model_path)

    print(f"Pipeline de modelo guardado en: {model_path}")

    # ruta del modelo vía XCom para la siguiente tarea
    return str(model_path)

# 4. Interfaz de Gradio 

def predict(file_obj, model_path):
    """
    Función auxiliar que carga el pipeline y realiza predicciones
    sobre un archivo JSON de entrada.
    """
    try:
        # modelo + preprocesador
        pipeline = joblib.load(model_path)

        # JSON desde el objeto de archivo temporal de Gradio
        # file_obj.name contiene la ruta al archivo temporal
        input_data = pd.read_json(file_obj.name, orient='records')

        # predicción
        predictions = pipeline.predict(input_data)

        print(f'Predicción bruta: {predictions}')

        # salida a etiquetas legibles
        labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

        # primera predicción
        return {'Predicción': labels[0]}

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return {'Error': str(e)}


def gradio_interface(**kwargs):
    """
    Lanza una interfaz de Gradio para interactuar con el modelo entrenado.

    Depende de:
    - XCom de la tarea 'preprocess_and_train'.
    """
    # la ruta del modelo desde la tarea de entrenamiento
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='preprocess_and_train')

    if model_path is None:
        print("Error: No se pudo obtener la ruta del modelo desde XCom.")
        return

    print(f"Cargando modelo desde: {model_path} para la interfaz de Gradio.")

    # función lambda para "congelar" el model_path
    prediction_fn = lambda file_obj: predict(file_obj, model_path)

    interface = gr.Interface(
        fn=prediction_fn,
        inputs=gr.File(label="Sube un archivo JSON con los datos del postulante"),
        outputs=gr.Json(label="Resultado de la Predicción"), 
        title="Predicción de Decisión de Contratación",
        description="Sube un archivo JSON con las características del postulante para predecir si será contratado o no."
    )

    print("Lanzando interfaz de Gradio en http://127.0.0.1:7860 (o similar)")
    interface.launch(share=True)