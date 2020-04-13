# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:25:43 2020

@author: allan

Utilizando árvore de decisão
"""

import pandas as pd

base = pd.read_csv(r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Decision_Tree\risco_credito.csv')

base.head()

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
# Realizando pre-processamento dos previsores que são dados categóricos pois
# Decision-Tree não utiliza dados categóricos
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

# Aplicando o algoritmo de Árvores de Decisão 
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)

print(classificador.feature_importances_)

export.export_graphviz(classificador,
                       out_file=r'C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\Decision_Tree\arvore.dot',
                       feature_names=['historia', 'divida',
                                       'garantias', 'renda'],
                       class_names=['alto', 'moderado', 'baixo'],
                       filled=True,
                       leaves_parallel=True)

# Realizando a classificação com um dado não incluso nos previsores
# historico bom, divida alta, garantia nenhuma e renda > 15
# historico ruim, divida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])

print(classificador.classes_)
