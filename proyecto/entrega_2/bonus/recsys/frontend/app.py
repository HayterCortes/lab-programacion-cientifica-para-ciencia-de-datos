"""
Frontend (Gradio) para el Sistema de Recomendación.
"""

import gradio as gr
import requests
import pandas as pd

# --- Configuración ---
#
# 'backend-recsys' será el nombre de nuestro servicio de Docker
BACKEND_URL = "http://backend-recsys:8000/recommendations/"

# --- Lógica de la Aplicación ---

def get_recommendations_for_customer(customer_id):
    """
    Llama a la API de FastAPI (RecSys) y formatea la respuesta.
    """
    if customer_id is None:
        return pd.DataFrame(columns=["product_id", "similarity_score"]), "Por favor, ingresa un ID de cliente."

    try:
        customer_id = int(customer_id)
        response = requests.get(f"{BACKEND_URL}{customer_id}")

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Error desconocido")
            return pd.DataFrame(columns=["product_id", "similarity_score"]), f"Error de la API: {error_detail}"

        data = response.json()
        recommendations = data.get("recommendations", [])

        if not recommendations:
            return pd.DataFrame(columns=["product_id", "similarity_score"]), "No se encontraron recomendaciones para este cliente."

        df = pd.DataFrame(recommendations)
        df = df.rename(columns={"product_id": "ID de Producto Recomendado", "similarity_score": "Puntaje de Similitud"})

        status_message = f"Recomendaciones para el Cliente {customer_id} (basado en {len(data.get('based_on_purchases', []))} productos comprados)"

        return df, status_message

    except requests.ConnectionError:
        return pd.DataFrame(columns=["product_id", "similarity_score"]), "Error: No se pudo conectar al backend de RecSys."
    except Exception as e:
        return pd.DataFrame(columns=["product_id", "similarity_score"]), f"Ocurrió un error inesperado: {e}"

# --- Definición de la Interfaz ---

with gr.Blocks(theme=gr.themes.Soft(), title="SodAI Drinks RecSys") as demo:
    gr.Markdown(
        """
        # SodAI Drinks - Sistema de Recomendación
        Este sistema utiliza un filtro colaborativo Item-Item.

        **Instrucciones de Uso:**
        1.  Ingresa un `customer_id` (ej. 25743, 25744, etc.).
        2.  Presiona "Generar Recomendaciones".
        3.  El sistema encontrará productos similares a los que este cliente ya ha comprado, 
            basado en los patrones de compra de todos los clientes.
        """
    )

    with gr.Row():
        customer_id_input = gr.Number(label="ID de Cliente (customer_id)")
        submit_button = gr.Button("Generar Recomendaciones")

    with gr.Column():
        status_output = gr.Textbox(label="Estado", interactive=False)
        results_output = gr.DataFrame(label="Recomendaciones (Top 5)")

    submit_button.click(
        fn=get_recommendations_for_customer,
        inputs=customer_id_input,
        outputs=[results_output, status_output]
    )

# Lanzar la aplicación
demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)