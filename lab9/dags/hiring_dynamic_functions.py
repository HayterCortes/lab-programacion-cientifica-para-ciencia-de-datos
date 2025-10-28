import pandas as pd
import joblib
from pathlib import Path
from glob import glob 

# Importaciones de Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#  1. Creación de Carpetas 

def create_folders(**kwargs):
    """
    Crea una estructura de carpetas basada en la fecha de ejecución (ds).
    Pasa las rutas de las carpetas creadas vía XCom.
    """
    # fecha de ejecución del contexto de Airflow
    execution_date = kwargs['ds']
    
    # ruta base en el directorio de trabajo actual
    base_path = Path.cwd() / execution_date
    
    # subcarpetas (incluyendo 'preprocessed')
    raw_path = base_path / 'raw'
    preprocessed_path = base_path / 'preprocessed' # Nueva carpeta
    splits_path = base_path / 'splits'
    models_path = base_path / 'models'
    
    folders_to_create = [
        base_path, raw_path, preprocessed_path, splits_path, models_path
    ]
    for path in folders_to_create:
        path.mkdir(parents=True, exist_ok=True)
        
    print(f"Carpetas creadas en: {base_path}")
    
    # diccionario con las rutas para que Airflow las pase por XCom
    return {
        'base_path': str(base_path),
        'raw_path': str(raw_path),
        'preprocessed_path': str(preprocessed_path),
        'splits_path': str(splits_path),
        'models_path': str(models_path)
    }

#  2. Carga y Fusión de Datos 

def load_and_merge(**kwargs):
    """
    Lee data_1.csv y (si existe) data_2.csv de la carpeta 'raw',
    los concatena y guarda el resultado en 'preprocessed'.
    
    Depende de:
    - XCom de la tarea 'create_folders'.
    - Archivos de datos en la carpeta 'raw'
    """
    #  rutas desde la tarea 'create_folders'
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='create_folders_task') 
    
    raw_path = Path(paths['raw_path'])
    preprocessed_path = Path(paths['preprocessed_path'])
    
    data_1_file = raw_path / 'data_1.csv'
    data_2_file = raw_path / 'data_2.csv'
    
    # Lista para almacenar los DataFrames a concatenar
    dfs_to_merge = []
    
    # Cargar data_1.csv 
    if data_1_file.exists():
        print(f"Leyendo {data_1_file}...")
        dfs_to_merge.append(pd.read_csv(data_1_file))
    else:
        raise FileNotFoundError(f"Archivo requerido {data_1_file} no encontrado.")

    # Cargar data_2.csv
    if data_2_file.exists():
        print(f"Leyendo {data_2_file}...")
        dfs_to_merge.append(pd.read_csv(data_2_file))
    else:
        print("data_2.csv no encontrado, se omitirá.")
        
    # Concatenar los DataFrames
    print("Concatenando archivos...")
    merged_df = pd.concat(dfs_to_merge, ignore_index=True)
    
    # Guardar en la carpeta 'preprocessed'
    output_file = preprocessed_path / 'merged_data.csv'
    merged_df.to_csv(output_file, index=False)
    
    print(f"Datos fusionados y guardados en: {output_file}")

#  3. División de Datos 

def split_data(**kwargs):
    """
    Lee los datos de 'preprocessed', aplica hold-out (80/20) estratificado
    y guarda los sets de train/test en 'splits'.
    
    Depende de:
    - XCom de la tarea 'create_folders_task'.
    - Archivo 'merged_data.csv' en la carpeta 'preprocessed'.
    """
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='create_folders_task') 
    
    preprocessed_path = Path(paths['preprocessed_path'])
    splits_path = Path(paths['splits_path'])
    
    input_file = preprocessed_path / 'merged_data.csv'
    
    print(f"Leyendo datos fusionados de: {input_file}")
    df = pd.read_csv(input_file)
    
    # features (X) y target (y)
    TARGET = 'HiringDecision'
    if TARGET not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET}' no se encuentra en los datos.")
        
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    # Split 80/20 estratificado con semilla
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42,  # Semilla fija
        stratify=y         # Mantener proporción
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

#  4. Entrenamiento de Modelo (Parametrizado) 

def train_model(model, model_name, **kwargs):
    """
    Función genérica para entrenar un modelo.
    Recibe una instancia del modelo y un nombre para guardarlo.
    
    Lee solo el set de entrenamiento de 'splits', define un pipeline,
    entrena y guarda el pipeline en 'models'.
    
    Args:
        model (sklearn.BaseEstimator): Instancia del clasificador (ej. RandomForestClassifier()).
        model_name (str): Nombre para el archivo (ej. "random_forest").
    
    Depende de:
    - XCom de la tarea 'create_folders_task'.
    - Archivo 'train.csv' en la carpeta 'splits'.
    """
    print(f" Iniciando entrenamiento para: {model_name} ")
    
    # rutas desde 'create_folders_task'
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='create_folders_task') 
    
    splits_path = Path(paths['splits_path'])
    models_path = Path(paths['models_path'])
    
    # solo los datos de entrenamiento
    print(f"Leyendo set de entrenamiento...")
    train_df = pd.read_csv(splits_path / 'train.csv')
    
    # features y target
    TARGET = 'HiringDecision'
    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]

    # Definición del Pipeline de Preprocesamiento 
    numeric_features = [
        'Age', 'ExperienceYears', 'PreviousCompanies', 
        'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore'
    ]
    categorical_features = ['Gender', 'EducationLevel', 'RecruitmentStrategy']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    #  Definición del Pipeline Completo 
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model) # Usamos el modelo pasado como argumento
    ])
    
    #  Entrenamiento 
    print(f"Entrenando pipeline con {model_name}...")
    pipeline.fit(X_train, y_train)
    print("Entrenamiento finalizado.")
    
    #  Guardado del Modelo 
    model_path = models_path / f'{model_name}.joblib'
    joblib.dump(pipeline, model_path)
    
    print(f"Pipeline de {model_name} guardado en: {model_path}")
    print(f" Finalizado entrenamiento para: {model_name} ")

#  5. Evaluación y Selección de Modelo 

def evaluate_models(**kwargs):
    """
    Lee todos los modelos entrenados de la carpeta 'models'
    y los evalúa contra el set de prueba de 'splits'.
    
    Selecciona el mejor modelo basado en 'accuracy', lo guarda
    como 'best_model.joblib' e imprime los resultados.
    
    Depende de:
    - XCom de la tarea 'create_folders_task'.
    - Archivo 'test.csv' en la carpeta 'splits'.
    - Archivos '.joblib' en la carpeta 'models' (generados por las tareas de 'train_model').
    """
    print(" Iniciando evaluación de modelos ")
    
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='create_folders_task') 
    
    splits_path = Path(paths['splits_path'])
    models_path = Path(paths['models_path'])
    
    # 1. Cargar el set de prueba
    print("Cargando set de prueba...")
    test_df = pd.read_csv(splits_path / 'test.csv')
    TARGET = 'HiringDecision'
    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]
    
    # 2. Encontrar todos los modelos entrenados
    model_files = [
        Path(f) for f in glob(str(models_path / '*.joblib')) 
        if Path(f).name != 'best_model.joblib'
    ]

    if not model_files:
        print("Error: No se encontraron modelos .joblib en la carpeta 'models'.")
        return

    print(f"Modelos encontrados: {[f.name for f in model_files]}")

    best_accuracy = -1.0
    best_model_pipeline = None
    best_model_name = ""

    # 3. Evaluar cada modelo
    for model_file in model_files:
        try:
            pipeline = joblib.load(model_file)
            model_name = model_file.stem # Obtiene el nombre del archivo sin extensión
            
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Modelo: {model_name} | Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_pipeline = pipeline
                best_model_name = model_name
                
        except Exception as e:
            print(f"Error al cargar o evaluar {model_file}: {e}")

    # 4. Seleccionar y guardar el mejor
    if best_model_pipeline:
        print(f"\n Selección Final ")
        print(f"Mejor Modelo: {best_model_name}")
        print(f"Mejor Accuracy: {best_accuracy:.4f}")
        
        best_model_path = models_path / 'best_model.joblib'
        joblib.dump(best_model_pipeline, best_model_path)
        
        print(f"Mejor modelo guardado en: {best_model_path}")
        
        # Devolver la ruta al mejor modelo para tareas futuras 
        return str(best_model_path)
    else:
        print("No se pudo seleccionar un mejor modelo.")