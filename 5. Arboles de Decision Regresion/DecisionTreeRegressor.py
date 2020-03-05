"""
Árboles de Decisión Regresión
@author: 
"""

#%% Importo librerías a utilizar

#Se importan la librerias a utilizar
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


#%% Preparación y comprensión de los datos

#Importamos los datos de la misma librería de scikit-learn
boston = datasets.load_boston()
print(boston)
print()

#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(boston.keys())
print()

#Verifico las características del dataset
print('Características del dataset:')
print(boston.DESCR)

#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(boston.data.shape)
print()

#Verifico la información de las columnas
print('Nombres columnas:')
print(boston.feature_names)

#Seleccionamos solamente la columna 6 del dataset
X_adr = boston.data[:, np.newaxis, 5]

#Defino los datos correspondientes a las etiquetas
y_adr = boston.target

#Graficamos los datos correspondientes
plt.scatter(X_adr, y_adr)
plt.show()


#%% Implementación del modelo de Arboles de Decisión Regresión

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_adr, y_adr, test_size=0.2)


#Defino el algoritmo a utilizar
adr = DecisionTreeRegressor(max_depth = 5)
#adr = DecisionTreeRegressor()


#Entreno el modelo
adr.fit(X_train, y_train)

#Realizo una predicción
Y_pred = adr.predict(X_test)


#%% Resultados del modelo

#Graficamos los datos de prueba junto con la predicción
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, adr.predict(X_grid), color='red', linewidth=3)
plt.show()
print('DATOS DEL MODELO ÁRBOLES DE DECISIÓN REGRESION')
print()
print('Precisión del modelo:')
print(adr.score(X_train, y_train))