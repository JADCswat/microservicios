import json

patterns = []
responses = []

# Preguntar por la etiqueta "tag"
tag = input("Ingresa la etiqueta para estas preguntas y respuestas: ")

while True:
    pregunta = input("Ingresa una pregunta o escribe 'salir' para finalizar el programa: ")
    if pregunta.lower() == "salir":
        break
    respuesta = input("Ingresa la respuesta a la pregunta: ")
    patterns.append(pregunta)
    responses.append(respuesta)

qa_pairs = {"tag": tag, "patterns": patterns, "responses": responses}

# Abrir el archivo existente en modo lectura y escritura
with open('intents.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
    # Agregar el nuevo par de valores a la lista de intenciones
    data['intents'].append(qa_pairs)
    # Regresar al inicio del archivo para no sobreescribir el contenido
    f.seek(0)
    # Escribir el objeto JSON al archivo con la codificación UTF-8 y ensure_ascii como False
    json.dump(data, f, ensure_ascii=False, indent=2)
    # Truncar el contenido restante del archivo si lo hubiera
    f.truncate()

print("Se ha agregado correctamente el par de preguntas y respuestas al archivo intents.json.")