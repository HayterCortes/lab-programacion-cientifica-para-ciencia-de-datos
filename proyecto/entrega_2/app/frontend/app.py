"""
Frontend (Gradio) para la aplicación SodAI Drinks.

Se conecta al backend de FastAPI para obtener y mostrar recomendaciones.
"""

import gradio as gr
import requests
import pandas as pd

# --- Configuración ---
BACKEND_URL = "http://backend:8000/recommendations/"

# --- Lógica de la Aplicación ---

def get_recommendations_for_customer(customer_id):
    """
    Llama a la API de FastAPI y formatea la respuesta.
    """
    if customer_id is None:
        return pd.DataFrame(columns=["product_id", "probability"]), "Por favor, ingresa un ID de cliente."

    try:
        # Convertir a int para asegurar que sea un ID válido
        customer_id = int(customer_id)

        # Llamar a la API
        response = requests.get(f"{BACKEND_URL}{customer_id}")

        # Manejar errores de la API (ej. 404 Cliente no encontrado)
        if response.status_code != 200:
            error_detail = response.json().get("detail", "Error desconocido")
            return pd.DataFrame(columns=["product_id", "probability"]), f"Error de la API: {error_detail}"

        # Éxito
        data = response.json()
        recommendations = data.get("recommendations", [])

        if not recommendations:
            return pd.DataFrame(columns=["product_id", "probability"]), "No se encontraron recomendaciones para este cliente."

        # Convertir a DataFrame para Gradio
        df = pd.DataFrame(recommendations)

        # Formatear probabilidad como porcentaje
        df['probability'] = (df['probability'] * 100).map('{:,.2f}%'.format)
        df = df.rename(columns={"product_id": "ID de Producto", "probability": "Probabilidad de Compra"})

        return df, f"Top 5 recomendaciones para el Cliente {customer_id}"

    except requests.ConnectionError:
        return pd.DataFrame(columns=["product_id", "probability"]), "Error: No se pudo conectar al backend de FastAPI. ¿Está corriendo?"
    except Exception as e:
        return pd.DataFrame(columns=["product_id", "probability"]), f"Ocurrió un error inesperado: {e}"

# --- Definición de la Interfaz ---

with gr.Blocks(theme=gr.themes.Soft(), title="SodAI Drinks Recommender") as demo:
    gr.Markdown(
        """
        # 🥤 SodAI Drinks - Sistema de Recomendación
        Bienvenido al sistema de predicción de compras.

        **Instrucciones de Uso:**
        1.  Ingresa un `customer_id` (ej. 25743, 25744, etc.).
        2.  Presiona "Obtener Recomendaciones".
        3.  El sistema te mostrará los 5 productos que este cliente tiene más probabilidades de comprar la próxima semana,
            basado en el último entrenamiento del modelo.
        """
    )

    with gr.Row():
        customer_id_input = gr.Number(label="ID de Cliente (customer_id)")
        submit_button = gr.Button("Obtener Recomendaciones")

    with gr.Column():
        status_output = gr.Textbox(label="Estado", interactive=False)
        results_output = gr.DataFrame(label="Recomendaciones de Productos")

    # Conectar los componentes
    submit_button.click(
        fn=get_recommendations_for_customer,
        inputs=customer_id_input,
        outputs=[results_output, status_output]
    )

# Lanzar la aplicación
demo.launch(
    server_name="0.0.0.0",  # Hacerlo accesible dentro de Docker
    server_port=7860
)