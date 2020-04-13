# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:00:21 2020

@author: allan
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, \
    mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\linear_regression\house_prices.csv')

""" Determinando as variáveis DEPENDENTES e EXPLANATÓRIOS """
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

X = X.reshape(-1, 1)

""" Divisão de teste e treino """
X_treinamento, X_teste, y_treinamento, y_teste = \
    train_test_split(X,
                     y,
                     test_size=0.3,
                     random_state=0)

regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

previsoes = regressor.predict(X_teste)

mae = mean_absolute_error(y_teste, previsoes)
regressor.score(X_teste, y_teste)

regressor.intercept_
len(regressor.coef_)
