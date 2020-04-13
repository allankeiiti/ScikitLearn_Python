# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:10:09 2020

@author: allan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Cross_Validation\credito.csv')

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

""" Utilizando o K-Fold """
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                   np.zeros(shape=(previsores.shape[0], 1))):
    # print(f'Indice treinamento: {indice_treinamento}')
    # print(f'Indice teste: {indice_teste}')
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

""" Matriz de Confusão """
matriz_final = np.mean(matrizes, axis=0)

""" Obtendo os resultados """
resultados = np.asarray(resultados)
resultados.mean()


resultados.std()