# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:13:35 2020

@author: allan
"""

import pandas as pd

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Naive_Bayes\credito.csv')

# Preprocessamento
base.loc[base.age < 0, 'age'] = 40.92
del base['clientid']

previsores = base.iloc[:, 0:3].values
classe = base.iloc[:, 3].values


from sklearn.preprocessing import Imputer
# Ajustando valores missing, substituindo-os pela média
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
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

# Iniciando a aplicação do Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

# Aplicando o Naive Bayes no previsores_teste, é visível seus acertos e erros
# comparando com o classe_teste
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
