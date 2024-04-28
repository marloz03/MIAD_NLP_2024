#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
from m10_model_deployment import predict_price
import joblib
from flask_cors import CORS
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='CAR price prediction API',
    description='API para predicción de precios de autos.')

ns = api.namespace('predict', 
     description='Predicción de precio de auto.')
   
parser = api.parser()

parser.add_argument(
    'YEAR', 
    type=int, 
    required=True, 
    help='Año del auto', 
    location='args')

parser.add_argument(
    'MILEAGE', 
    type=int, 
    required=True, 
    help='Kilometraje del auto', 
    location='args')

parser.add_argument(
    'STATE', 
    type=str, 
    required=True, 
    help='Código del estado donde esta registrado el auto', 
    location='args')

parser.add_argument(
    'MAKE', 
    type=str, 
    required=True, 
    help='Empresa que manufacturó el auto', 
    location='args')

parser.add_argument(
    'MODEL', 
    type=str, 
    required=True, 
    help='Modelo del auto', 
    location='args')


resource_fields = api.model('Resource', {
    'precio estimado': fields.String,
})

@ns.route('/')
class CarPriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_price(args['YEAR'], args['MILEAGE'], args['STATE'], args['MAKE'], args['MODEL'], model_mapping, encoder)
        }, 200
    
    
if __name__ == '__main__':

    #Cargo la data de entrenamiento para el mapping
    dataTraining = pd.read_csv('dataTrain_carListings.zip')
    model_means = dataTraining.groupby('Model')['Price'].mean()
    model_mapping = model_means.to_dict()

    #Extraigo el encoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(dataTraining[['State', 'Make']])

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
