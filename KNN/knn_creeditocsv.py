# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:54:26 2020

@author: allan
"""

import pandas as pd

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\KNN\credito.csv')

# Preprocessamento
base.loc[base.age < 0, 'age'] = 40.92
del base['clientid']

previsores = base.iloc[:, 0:3].values
classe = base.iloc[:, 3].values


from sklearn.impute import SimpleImputer
# Ajustando valores missing, substituindo-os pela média
imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Ajustando a escala dos atributos, para evitar problemas no algoritmo
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão de ROWs de teste e treino
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = \
    train_test_split(previsores,
                     classe,
                     test_size=0.25,
                     random_state = 0)

# Aplicando o KNN
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Criando o Score de acurácia e matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(classe_teste)
