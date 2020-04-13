# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:39:09 2020

@author: allan
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
# Este CSV contém o dados do risco_credito.csv, porém desconsiderando
# rows com risco "médio" por escolha do instrutor para fácil interpretação.
base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Log_Regression\risco_credito2.csv')

# Indicando quais são os atributos previsores e classe
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

""" Pre-Processamento dos dados """

# Realizando o pre-processamento dos dados do dataset transformando os dados
# categóricos em números
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

""" Implementando a Regressão Linear Logistica """

# Detalhe que o parâmetro SOLVER é indicado qual algoritmo a ser utilizado para
# descobrir o coeficiente da regressão logística
# MAX_ITER indica quantas vezes será feita o ajuste dos coeficientes
classificador = LogisticRegression(solver=lbfgs, max_iter=100)
classificador.fit(x=previsores, y=classe)

# Coeficiente B0
print(classificador.intercept_)

# Demais Coeficientes
print(classificador.conf_)

""" Realizando o teste de classificação """

# historico boa, divida alta, garantias nenhuma, renda > 35
# historico ruim, divida alta, garantias adequada, renda < 15
resultado = classificador.predict([0,0,1,2], [3,0,0,0])
print(resultado)

# Verificando as probabilidades
resultado_2 = classificador.predict_proba([0,0,1,2], [3,0,0,0])
print(resultado_2)
