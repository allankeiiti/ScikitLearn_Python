# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:40:12 2020

@author: allan
"""

import pandas as pd

base = pd.read_csv(r'C:\\Users\\allan\\Documents\Python_Machine_Learning_Jones_Granatyr\censo.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Substituindo valores categóricos por inteiros
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
# labels = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

# Transformando variáveis em Dummy
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
previsores = one_hot_encoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe).toarray()