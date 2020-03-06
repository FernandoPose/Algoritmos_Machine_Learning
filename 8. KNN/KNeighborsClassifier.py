"""
K Vecinos mรกs Cercanos
@author: 
"""

#%% Importo librerías a utilizar

#Se importan la librerias a utilizar
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score


#%% Preparación y comprensión de los datos

#Importamos los datos de la misma librerรญa de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)

#Verifico la informaciรณn contenida en el dataset
print('Informaciรณn en el dataset:')
print(dataset.keys())
print()

#Verifico las caracterรญsticas del dataset
print('Caracterรญsticas del dataset:')
print(dataset.DESCR)

#Seleccionamos todas las columnas
X = dataset.data

#Defino los datos correspondientes a las etiquetas
y = dataset.target


#%% Implementación del modelo de K Vecinos más Cercanos

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Defino el algoritmo a utilizar. Los datos en los argumentos son los mismos que vienen por default
algoritmo = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)  

#Entreno el modelo
algoritmo.fit(X_train, y_train)

#Realizo una predicciรณn
y_pred = algoritmo.predict(X_test)

#Verifico la matriz de Confusiรณn
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusiรณn:')
print(matriz)

#Calculo la precisiรณn del modelo
precision = precision_score(y_test, y_pred)
print('Precisiรณn del modelo:')
print(precision)