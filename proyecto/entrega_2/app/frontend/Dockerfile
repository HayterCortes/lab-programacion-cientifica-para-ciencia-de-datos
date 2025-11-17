# Usar una imagen de Python ligera
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del frontend (app.py)
COPY . .

# Exponer el puerto en el que correrá Gradio
EXPOSE 7860

# Comando para iniciar la aplicación
CMD ["python", "app.py"]