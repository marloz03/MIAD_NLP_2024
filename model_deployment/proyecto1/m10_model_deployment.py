#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import OneHotEncoder

def predict_price(year, mileage, state, make, model, mapping, encoder) :

    #Cargo el modelo para predecir
    rgr = joblib.load(os.path.dirname(__file__) + '/car_price.pkl')

    dataTesting = pd.DataFrame([[year, mileage, state, make, model]], columns=['Year', 'Mileage', 'State', 'Make', 'Model'])

    dataTesting['Model_Encoded'] = dataTesting['Model'].map(mapping)
    encoded_features = encoder.transform(dataTesting[['State', 'Make']])
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(['State', 'Make']))
    dataTesting_encoded = pd.concat([dataTesting, encoded_df], axis=1)
    dataTesting_encoded.drop(['State', 'Make', 'Model'], axis=1, inplace=True)

    # Make prediction
    y_test_pred = round(rgr.predict(dataTesting_encoded)[0], 2)

    return y_test_pred


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor agregue los siguientes datos: Year, Mileage, State, Make, Model')
        
    else:
        #Cargo la data de entrenamiento para el mapping
        dataTraining = pd.read_csv('dataTrain_carListings.zip')
        model_means = dataTraining.groupby('Model')['Price'].mean()
        model_mapping = model_means.to_dict()

        #Extraigo el encoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(dataTraining[['State', 'Make']])

        year= sys.argv[1]
        mileage= sys.argv[2]
        state= sys.argv[3]
        make=sys.argv[4]
        model=sys.argv[5]
        mapping = model_mapping

        precio_estimado = predict_price(year, mileage, state, make, model, mapping, encoder)
        
        print('Precio estimado: ', precio_estimado)
        