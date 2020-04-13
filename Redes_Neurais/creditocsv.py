# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:01:23 2020

@author: allan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Redes_Neurais\credito.csv')

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

""" Utilizando Redes Neurais """
from sklearn.neural_network import MLPClassifier

classificador = MLPClassifier(verbose=True,
                              max_iter=1000,
                              tol=0.000010,
                              solver='adam',
                              hidden_layer_sizes=(100),
                              activation='relu',
                              )
# Passando verbose True, ele printa o valor de erro/loss a cada época
classificador.fit(previsores_treinamento, classe_treinamento)
previsores = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)