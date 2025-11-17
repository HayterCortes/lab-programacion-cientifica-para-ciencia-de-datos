# airflow/scripts/config.py

# Columnas numéricas para el preprocesador
NUMERIC_FEATURES = [
    'lag_1_purchased', 'lag_1_items', 'lag_2_purchased', 'lag_2_items',
    'lag_3_purchased', 'lag_3_items', 'lag_4_purchased', 'lag_4_items',
    'rolling_4w_sum_purchased', 'rolling_4w_mean_purchased', 
    'rolling_4w_sum_items', 'rolling_4w_mean_items',
    'rolling_8w_sum_purchased', 'rolling_8w_mean_purchased', 
    'rolling_8w_sum_items', 'rolling_8w_mean_items',
    'week_of_year', 'month'
] 

# Columnas categóricas para el preprocesador
CATEGORICAL_FEATURES = [
    'brand', 'package', 'sub_category', 'customer_type'
] 

# Columnas identificadoras
ID_COLS = ['customer_id', 'product_id', 'target_week']

# Variable objetivo
TARGET_COL = 'target'