from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Cargar el modelo y el tokenizer preentrenados para emociones en español
modelo_nombre = "mrm8488/bert-spanish-emotion"
tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
modelo = AutoModelForSequenceClassification.from_pretrained(modelo_nombre)

# Crear el pipeline de análisis de emociones
analizador_emociones = pipeline("text-classification", model=modelo, tokenizer=tokenizer, return_all_scores=True)

# Función para mapear emociones a retroalimentación
def proporcionar_retroalimentacion(emociones):
    retroalimentacion = []
    for emocion in emociones:
        etiqueta = emocion['label']
        puntaje = emocion['score']
        if puntaje < 0.5:
            continue  # Ignorar emociones con baja probabilidad
        mensaje = ""
        if etiqueta == "alegría":
            mensaje = "¡Nos alegra saber que te sientes feliz! 😊"
        elif etiqueta == "tristeza":
            mensaje = "Lamentamos que te sientas triste. Estamos aquí para ayudarte. 😔"
        elif etiqueta == "ira":
            mensaje = "Sentimos que te sientas enfadado. ¿Cómo podemos mejorar? 😠"
        elif etiqueta == "miedo":
            mensaje = "Entendemos que puedas sentir miedo. Queremos apoyarte. 😨"
        elif etiqueta == "amor":
            mensaje = "¡Es maravilloso sentir amor! 💖"
        elif etiqueta == "sorpresa":
            mensaje = "¡Vaya, parece que algo te sorprendió! 😲"
        # Agrega más emociones y retroalimentaciones según sea necesario
        if mensaje:
            retroalimentacion.append(mensaje)
    return retroalimentacion

# Función principal para analizar texto y proporcionar retroalimentación
def analizar_texto(texto):
    resultados = analizador_emociones(texto)
    emociones_detectadas = resultados[0]  # El pipeline devuelve una lista de listas
    retroalimentacion = proporcionar_retroalimentacion(emociones_detectadas)
    return emociones_detectadas, retroalimentacion

# Ejemplo de uso
if __name__ == "__main__":
    texto = input("Introduce el texto a analizar: ")
    emociones, feedback = analizar_texto(texto)
    
    print("\nEmociones detectadas:")
    for emocion in emociones:
        print(f"{emocion['label']}: {emocion['score']:.2f}")
    
    print("\nRetroalimentación:")
    for mensaje in feedback:
        print(mensaje)
