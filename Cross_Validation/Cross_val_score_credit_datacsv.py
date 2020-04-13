# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:45:04 2020

@author: allan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
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

""" Criando o Classificador """
classificador = GaussianNB()

""" Utilizando o CROSS VALIDATION, onde CV é o K """
resultados = cross_val_score(classificador, previsores, classe, cv=10)

""" Obtendo o resultado dos testes """
resultados.mean()

""" Obtendo o desvio padrão """
# Valores de desvio padrão pode indicar OverFitting
resultados.std()
