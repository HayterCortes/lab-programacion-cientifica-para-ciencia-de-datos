"""
Frontend (Gradio) para el Chatbot Conversacional (Bonus LLM).
Implementa gr.ChatInterface con manejo de historial.
"""

import gradio as gr
import requests
import time

# --- Configuración ---
BACKEND_URL = "http://backend-llm:8000/chat"

# --- Lógica de la Aplicación ---

def chat_with_bot(message, history):
    """
    Envía el mensaje Y el historial al backend de FastAPI.
    """
    # Formatear el historial para Pydantic
    history_formatted = []
    for user_msg, bot_msg in history:
        history_formatted.append({"type": "human", "content": user_msg})
        if bot_msg:
            history_formatted.append({"type": "ai", "content": bot_msg})

    try:
        payload = {
            "message": message,
            "history": history_formatted
        }

        response = requests.post(BACKEND_URL, json=payload)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Error desconocido")
            return f"Error del Backend: {error_detail}"

        data = response.json()
        response_text = data.get("response", "No se recibió respuesta del bot.")

        # Simular "streaming" 
        for i in range(len(response_text)):
            time.sleep(0.01)
            yield response_text[: i+1]

    except requests.ConnectionError:
        yield "Error Crítico: No se pudo conectar al backend del Chatbot."
    except Exception as e:
        yield f"Ocurrió un error inesperado: {e}"

# --- Definición de la Interfaz ---

demo = gr.ChatInterface(
    fn=chat_with_bot,
    title="SodAI Drinks - Chatbot de Datos", 
    description="Hola! Soy un agente de IA que puede consultar los datos de clientes y transacciones. (Cumple con Lab11)", # [cite: 694]
    examples=[
        "¿Cuántos clientes únicos hay?",
        "¿Cuántos productos únicos se encuentran en los datos?",
        "¿Cuántas transacciones ha realizado el cliente 25743?",
        "¿Cuál es el tipo de cliente (customer_type) del cliente 25743?"
    ],
    theme="soft" 
)

# Lanzar la aplicación
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False 
)