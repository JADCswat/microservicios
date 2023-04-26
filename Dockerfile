# Definir la imagen base
FROM python:alpine

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la aplicación en la imagen
COPY main.py /app/
COPY requirements.txt /app/

# Instalar las dependencias de la aplicación
RUN pip install -r requirements.txt

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 8080

# Ejecutar la aplicación cuando se inicie el contenedor
CMD ["python", "main.py"]

