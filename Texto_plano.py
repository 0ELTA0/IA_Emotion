import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Descargar el léxico de VADER
nltk.download('vader_lexicon')

# Crear el analizador de sentimientos
analyzer = SentimentIntensityAnalyzer()

while True:
    
    # Texto para analizar
    texto = input("Enter a sentence: ")
    
    # Analizar el texto
    resultados = analyzer.polarity_scores(texto)

    # Mostrar resultados
    print("Resultados del análisis de sentimientos:")
    print(f"Negatividad: {resultados['neg']}")
    print(f"Neutralidad: {resultados['neu']}")
    print(f"Positividad: {resultados['pos']}")
    print(f"Sentimiento compuesto: {resultados['compound']}")
