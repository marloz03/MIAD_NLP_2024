#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os




def predict_class(plot, tfidf_vectorizer) :

    #Cargo el modelo para predecir
    clf = joblib.load(os.path.dirname(__file__) + '/clasificador.pkl')
    
    data_testing = pd.DataFrame([[plot]], columns=['plot'])
    
    X_test_tfidf = tfidf_vectorizer.transform(data_testing['plot'])
    
    cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    df_respuesta = pd.DataFrame(clf.predict(X_test_tfidf), columns = cols)
    respuesta_final = ''

    for categoria, valor in df_respuesta.items():
        if valor[0] == 1 and respuesta_final == '':
            respuesta_final += str(categoria)
        elif valor[0] == 1 and respuesta_final != '':
            respuesta_final += ', ' + str(categoria)
    
    if respuesta_final == '':
        respuesta_final = 'El clasificador no encontró ningún género para esta película, por favor intente con otro plot.'

    return respuesta_final
        