# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:52:16 2020

@author: allan
"""

import pandas as pd

# CSV Composto pelas Colunas:
# ClientId, income, age, loan (Empréstimo), default
base = pd.read_csv(r"C:\Users\allan\Documents\Python_Machine_Learning_Jones_Granatyr\original.csv")

# Apaga ROWs cuja a 'age' é negativo
base.drop(base[base.age < 0].index, inplace=True)

# Preencher os valores com a média das idades corretas
base.age[base.age > 0].mean()

base.loc[base.age < 0, 'age'] = 40.92

pd.isnull(base['age']) # Pode ser (base.age) também

base.loc[base.age.isnull()]

# Dividindo o DataFrame em previsores e classes
# : Todas as linhas
# 1:4 da coluna 1 a coluna 4
previsores = base.iloc[:, 1:4].values

classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer = imputer.fit(previsores[:, 0:3])

previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])