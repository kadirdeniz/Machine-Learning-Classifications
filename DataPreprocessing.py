# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:44:36 2020

@author: deniz
"""

import pandas as pd

df = pd.read_csv('satislar.csv')
print(df)
print(df.isnull().sum())#Hangi column'da kaç tane boş değer oldugunu yazdırır
print(df.values)
df.dropna(axis=0)

from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit_transform(df.values)

tshirts= pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']])
tshirts.columns = ['color', 'size', 'price', 'classlabel']

'''DataFrame contains a nominal feature (color),
an ordinal feature (size), and a numerical feature
(price) column. '''

size_mapping = {'XL': 3, #Amele İsi Encoder kullanıcaz sakinn....
                'L': 2,
                'M': 1}

tshirts['size'] = tshirts['size'].map(size_mapping)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X=tshirts[['color','size','price']].values
X[:,0]=le.fit_transform(X[:,0])
print("kadir",X)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[0])
Y=ohe.fit_transform(X).toarray()

tshirts2=pd.get_dummies(tshirts[['color', 'size', 'price']])

'''
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
Dursun Burada
'''






