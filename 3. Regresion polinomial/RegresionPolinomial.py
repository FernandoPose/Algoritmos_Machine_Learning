"""
Regresi贸n Polinomial
@author: 
"""

#%% Importo librer韆s a utilizar

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


#%% Preparaci髇 y comprensi髇 de los datos

#Importamos los datos de la misma librer铆a de scikit-learn
boston = datasets.load_boston()
print(boston)
print()

#Verifico la informaci贸n contenida en el dataset
print('Informaci贸n en el dataset:')
print(boston.keys())
print()

#Verifico las caracter铆sticas del dataset
print('Caracter铆sticas del dataset:')
print(boston.DESCR)

#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(boston.data.shape)
print()

#Verifico la informaci贸n de las columnas
print('Nombres columnas:')
print(boston.feature_names)

#Seleccionamos solamente la columna 6 del dataset
X_p = boston.data[:, np.newaxis, 5]

#Defino los datos correspondientes a las etiquetas
y_p = boston.target

#Graficamos los datos correspondientes
plt.scatter(X_p, y_p)
plt.show()


#%% Implementaci髇 del modelo de Regresi髇 Polinomial

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2)

#Se define el grado del polinomio
poli_reg = PolynomialFeatures(degree = 2)

#Se transforma las caracter铆sticas existentes en caracter铆sticas de mayor grado
X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli = poli_reg.fit_transform(X_test_p)

#Defino el algoritmo a utilizar
pr = linear_model.LinearRegression()

#Entreno el modelo
pr.fit(X_train_poli, y_train_p)

#Realizo una predicci贸n
Y_pred_pr = pr.predict(X_test_poli)


#%% Resultados del modelo

#Graficamos los datos junto con el modelo
plt.scatter(X_test_p, y_test_p)
plt.plot(X_test_p, Y_pred_pr, color='red', linewidth=3)
plt.show()
print()
print('DATOS DEL MODELO REGRESI脫N POLINOMIAL')
print()
print('Valor de la pendiente o coeficiente "a":')
print(pr.coef_)
print('Valor de la intersecci贸n o coeficiente "b":')
print(pr.intercept_)
print('Precisi贸n del modelo:')
print(pr.score(X_train_poli, y_train_p))