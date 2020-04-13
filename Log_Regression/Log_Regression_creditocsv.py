# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:02:21 2020

@author: allan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Log_Regression\credito.csv')

""" Pré-Processamento dos dados """
# Removendo coluna clientid
del base['clientid']

# Substituindo idades negativas, o valor 40.92 foi obtido realizando a média
# das idades do dataset, desconsiderando as idades negativas, execute a linha
# abaixo para obter este valor

# base.loc[base.age >= 0].mean().age

base.loc[base.age < 0, 'age'] = 40.92

# Indicando atributos previsores e classe
previsores = base.iloc[:, 0:3].values
classe = base.iloc[:, 3].values


""" Ajustando valores missing """
imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

""" Ajustando escala dos atributos """
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

""" Divisão de teste e treino """
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = \
    train_test_split(previsores,
                     classe,
                     test_size=0.25,
                     random_state=0)
    
""" Criando o classificador """
classificador = LogisticRegression(random_state=1)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Criando o Score de acurácia e matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(classe_teste)
