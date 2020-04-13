# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:39:37 2020

@author: allan
"""

import pandas as pd

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Naive_Bayes\risco_credito.csv')

base.head()

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
# Realizando pre-processamento dos previsores que são dados categóricos
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

# Previsores - Vetor de treinamento
# Classe - Vetor Alvo
# Naive Bayes não aceita ATRIBUTOS categóricos
classificador.fit(previsores, classe)

# Realizando a classificação com um dado não incluso nos previsores
# historico bom, divida alta, garantia nenhuma e renda > 15
# historico ruim, divida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
