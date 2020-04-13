# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:40:29 2020

@author: allan
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Redes_Neurais\censo.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Substituindo valores categóricos por inteiros
labelencoder_previsores = LabelEncoder()

# Transformando dados categóricos em discreto
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

# Transformando variáveis em Dummy
one_hot_encoder = OneHotEncoder()
one_hot_encoder = OneHotEncoder(categories=[1, 3, 5, 6, 7, 8, 9, 13])
previsores = one_hot_encoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)


scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


""" Divisão de dados de treino e teste """
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = \
    train_test_split(previsores,
                     classe,
                     test_size=0.25,
                     random_state = 0)

""" Utilizando Redes Neurais """
from sklearn.neural_network import MLPClassifier

classificador = MLPClassifier(verbose=True,
                              max_iter=1000,
                              tol=0.000010)
classificador.fit(previsores_treinamento, classe_treinamento)


previsores = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)