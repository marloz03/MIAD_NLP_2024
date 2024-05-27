#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
from m10_model_deployment import predict_class
import joblib
from flask_cors import CORS
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer



# Cargar el modelo de lenguaje de spaCy
nlp = spacy.load("en_core_web_sm")

# Función de preprocesamiento para eliminar caracteres especiales, espacios adicionales y palabras cortas
def preprocess_text(text):
    # Eliminar caracteres especiales y dejar solo letras y espacios
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Eliminar espacios adicionales
    text = re.sub(r'\s+', ' ', text).strip()
    # Eliminar palabras con 3 o menos caracteres
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text

# Función para dar más peso a los adjetivos
def increase_weight_adjectives(text):
    doc = nlp(text)
    words = []
    for token in doc:
        if token.pos_ == 'ADJ':
            words.extend([token.text] * 2)  # Aumentar el peso de los adjetivos
        else:
            words.append(token.text)
    return ' '.join(words)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Clasificación de género de películas',
    description='API para clasificación de género de películas dado su plot.')

ns = api.namespace('predict', 
     description='Clasificación')
   
parser = api.parser()

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Plot de la película', 
    location='args')


resource_fields = api.model('Resource', {
    'Clasificación:': fields.String,
})

@ns.route('/')
class MovieClassApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot_procesado = increase_weight_adjectives(preprocess_text(args['plot']))
        return {
         "Clasificacion:": predict_class(plot_procesado, tfidf_vectorizer)
        }, 200
    
    
if __name__ == '__main__':

    # Cargar los archivos
    data_training = pd.read_csv('dataTraining.csv', header=None)
    
    # Eliminar la primera fila (encabezado) de dataTraining y dataTesting
    data_training = data_training.iloc[1:].reset_index(drop=True)
    
    # Asignar nombres a las columnas
    data_training.columns = ['ID', 'year', 'title', 'plot', 'genres', 'rating']
    
    print("Preprocesando texto...")
    # Aplicar preprocesamiento a la columna 'plot'
    data_training['plot'] = data_training['plot'].apply(preprocess_text)
    
    # Dar más peso a los adjetivos en la columna 'plot'
    data_training['plot'] = data_training['plot'].apply(increase_weight_adjectives)
    
    # Calcular la longitud promedio de las palabras en 'plot'
    data_training['plot_length'] = data_training['plot'].apply(lambda x: len(x.split()))
    average_plot_length = data_training['plot_length'].mean()
    threshold_plot_length = average_plot_length + 318

    # Filtrar los datos según la longitud de 'plot'
    filtered_data_training = data_training[data_training['plot_length'] <= threshold_plot_length]

    print("Cargando vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=200000, stop_words='english', token_pattern=r'\b[A-Za-z]+\b')
    X_train_tfidf = tfidf_vectorizer.fit_transform(filtered_data_training['plot'])

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
