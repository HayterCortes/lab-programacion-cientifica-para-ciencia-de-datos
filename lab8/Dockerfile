# 1. Imagen Base
# Usamos una imagen 'slim' para que sea ligera
FROM python:3.9-slim

# 2. Directorio de Trabajo
# Establece el directorio donde se ejecutará la app dentro del contenedor
WORKDIR /app

# 3. Copiar y instalar dependencias
# Copia solo el requirements.txt primero para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el código de la aplicación
# Copia solo main.py 
COPY main.py .

# 5. Exponer el puerto 
# Informa a Docker que el contenedor escuchará en el puerto 8000
EXPOSE 8000

# 6. Comando de ejecución
# Ejecuta la API usando uvicorn. 
# Usamos 0.0.0.0 para que sea accesible desde fuera del contenedor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]