from fastapi import FastAPI, Request, Form
from datetime import time
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import np as np
import json
import string
import time
import numpy as np
import unicodedata
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Desactiva el uso de instrucciones AVX


# Cargar datos del archivo JSON
with open("intents.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

# Preprocesamiento de los datos
entrenamiento = []
clases = []
documentos = []
ignorar = ["?", "!", ".", ","]

for intent in datos["intents"]:
    for patron in intent["patterns"]:
        # Convertir a minúsculas y eliminar signos de puntuación
        palabras = [palabra.lower() for palabra in patron.split() if palabra not in ignorar]
        entrenamiento.append(" ".join(palabras))
        clases.append(intent["tag"])
        documentos.append((palabras, intent["tag"]))

# Crear diccionario de palabras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(entrenamiento)
palabras = tokenizer.word_index
num_palabras = len(palabras) + 1

# Crear datos de entrenamiento
entradas = []
salidas = []
for doc in documentos:
    # Crear vectores one-hot para la entrada
    entrada = [0] * num_palabras
    for palabra in doc[0]:
        if palabra in palabras:
            entrada[palabras[palabra]] = 1
    entradas.append(entrada)

    # Crear vectores one-hot para la salida
    salida = [0] * len(clases)
    salida[clases.index(doc[1])] = 1
    salidas.append(salida)

# Convertir a matrices numpy
X = np.array(entradas)
Y = np.array(salidas)

# Definir modelo de red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=num_palabras, activation='relu'),
    tf.keras.layers.Dense(len(clases), activation='softmax')
])
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar modelo
modelo.fit(X, Y, epochs=200, batch_size=64, verbose=1)

#guardar modelo
modelo.save('modelo_chatbot_final.h5')

# cargar modelo
modelo = tf.keras.models.load_model('modelo_chatbot_final.h5')

def procesar_entrada(entrada):
    # Eliminar signos de puntuación y tildes
    entrada = entrada.lower()
    entrada = entrada.translate(str.maketrans('', '', string.punctuation))
    entrada = unicodedata.normalize('NFKD', entrada).encode('ASCII', 'ignore').decode('utf-8')
    return entrada


def chatbot_respuesta(texto: str) -> str:
    global response_time
    start_time = time.time()

    texto = procesar_entrada(texto)

    entrada = [0] * num_palabras
    palabras_entrada = [palabra.lower() for palabra in texto.split() if palabra not in ignorar]
    for palabra in palabras_entrada:
        if palabra in palabras:
            entrada[palabras[palabra]] = 1

    prediccion = modelo.predict(np.array([entrada]))
    respuesta_index = np.argmax(prediccion)
    tag_respuesta = clases[respuesta_index]

    preguntas_respuestas = {}
    for intent in datos["intents"]:
        if intent["tag"] == tag_respuesta:
            patterns = intent["patterns"]
            responses = intent["responses"]
            if len(patterns) == len(responses):
                for i, pattern in enumerate(patterns):
                    pregunta = procesar_entrada(pattern)
                    respuesta = responses[i]
                    preguntas_respuestas[pregunta] = respuesta

    if preguntas_respuestas:
        respuesta = preguntas_respuestas.get(texto.lower(), "")
        if respuesta == "":
            respuesta = np.random.choice(list(preguntas_respuestas.values()))
    else:
        respuesta = np.random.choice(responses)

    end_time = time.time()
    response_time = end_time - start_time
    response_time = f"{(end_time - start_time) * 1000:.2f} ms"
    return respuesta


app = FastAPI()
templates = Jinja2Templates(directory="templates")

messages = []


@app.get("/", response_class=HTMLResponse)
def read_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "messages": messages})


@app.post("/chat")
async def chatbot(request: Request, message: str = Form(...)):
    respuesta = chatbot_respuesta(message)
    messages.append(f"Usuario: {message}")
    messages.append(f"Chatbot: {respuesta}")
    return templates.TemplateResponse("index.html", {"request": request, "messages": messages, "response_time": response_time})