#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backend (FastAPI) para el Chatbot Conversacional (Bonus LLM).

Implementa un Agente LangChain (Pandas DataFrame Agent)
con memoria, para responder
preguntas sobre los datos de SodAI Drinks.
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from typing import List

# Importaciones de LangChain 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

# --- Configuración ---

# Rutas de datos dentro del contenedor
TRANS_PATH = "/app/data/transacciones.parquet"
CLIENTS_PATH = "/app/data/clientes.parquet"

# "Base de datos" en memoria para los dataframes
data_cache = {
    "df_trans": pd.DataFrame(),
    "df_clients": pd.DataFrame(),
}

# "Cerebro" del agente
agent_executor = None

# Memoria de Chat 
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Modelos Pydantic para la entrada/salida ---
class ChatHistoryItem(BaseModel):
    type: str
    content: str

class ChatMessage(BaseModel):
    message: str
    history: List[ChatHistoryItem] = []

# --- Funciones de "Entrenamiento" (Carga y Creación del Agente) ---

def load_and_create_agent():
    """
    Carga los DataFrames y crea el Agente de Pandas.
    """
    global agent_executor

    if not os.path.exists(TRANS_PATH) or not os.path.exists(CLIENTS_PATH):
        print(f"ADVERTENCIA: No se encontraron los archivos de datos. El Chatbot no funcionará.")
        return

    print("Cargando DataFrames para el Agente Chatbot...")
    data_cache["df_trans"] = pd.read_parquet(TRANS_PATH)
    data_cache["df_clients"] = pd.read_parquet(CLIENTS_PATH)

    print("DataFrames cargados. Creando Agente LLM...")

    # 1. Inicializar el LLM 
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    # 2. Crear el Agente de Pandas
    # Le damos al LLM acceso a ambos dataframes y sus nombres
    agent_executor = create_pandas_dataframe_agent(
        llm=llm,
        df=[data_cache["df_trans"], data_cache["df_clients"]],
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, # Para que podamos ver lo que "piensa" en los logs
        agent_executor_kwargs={"memory": chat_memory} 
    )
    print("Agente LLM con memoria creado y listo.")

# --- Eventos de Ciclo de Vida de la App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga los datos y el agente al iniciar la aplicación.
    """
    print("Iniciando aplicación Chatbot...")
    # Cargar la API Key desde el entorno (pasada por docker-compose)
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY no encontrada. El agente no funcionará.")
    else:
        print("GOOGLE_API_KEY encontrada.")

    load_and_create_agent()
    yield
    # Limpieza
    print("Apagando aplicación Chatbot...")
    data_cache.clear()

# --- Creación de la App ---

app = FastAPI(
    title="SodAI Drinks - Chatbot",
    description="API que usa un Agente LangChain para responder preguntas sobre los datos.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Endpoints de la API ---

@app.post("/chat")
def post_chat_message(chat_message: ChatMessage):
    """
    Recibe un mensaje del chat, lo procesa con el Agente y devuelve la respuesta.
    """
    if agent_executor is None:
        raise HTTPException(status_code=503, 
                            detail="El Agente LLM no está inicializado. ¿Falta la API Key o los archivos de datos?")

    print(f"Recibido: {chat_message.message}")

    # Cargar el historial de chat de Gradio en la memoria de LangChain
    chat_memory.clear() # Limpiar memoria anterior
    for item in chat_message.history:
        if item.type == "human":
            chat_memory.chat_history.add_user_message(item.content)
        elif item.type == "ai":
            chat_memory.chat_history.add_ai_message(item.content)

    try:
        # Invocar al Agente 
        response = agent_executor.invoke(chat_message.message)
        return {"response": response["output"]}

    except Exception as e:
        print(f"Error al invocar el agente: {e}")
        raise HTTPException(status_code=500, detail=str(e))