# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:32:24 2020

@author: allan
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot
base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\linear_regression\plano_saude.csv')

""" Definindo variáveis explanatórias e dependentes """
X = base.iloc[:, 0].values
y = base.iloc[:, 1].values

""" Obtendo a correlação de X e Y """
correlacao = np.corrcoef(X, y)

X = X.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X, y)

# B1
regressor.coef_
# B0
regressor.intercept_

""" Plottando """
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.title("Regressão Linear Simples")
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao1 = regressor.predict(40)
previsao2 = regressor.intercept_ + regressor.coef_ * 40

score = regressor.score(X, y)

visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()