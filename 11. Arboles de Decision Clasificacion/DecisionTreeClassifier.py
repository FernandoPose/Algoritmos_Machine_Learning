"""
Naive Bayes
@author: 
"""

#%% Importo librerías a utilizar

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score


#%% Preparación y comprensión de los datos

#Importamos los datos de la misma librería de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)

#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(dataset.keys())
print()

#Verifico las características del dataset
print('Características del dataset:')
print(dataset.DESCR)

#Seleccionamos todas las columnas
X = dataset.data

#Defino los datos correspondientes a las etiquetas
y = dataset.target

#%% Implementación del modelo de Árboles de decisión Clasificación

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Defino el algoritmo a utilizar
algoritmo = DecisionTreeClassifier(criterion = 'entropy')

#Entreno el modelo
algoritmo.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo.predict(X_test)

#Verifico la matriz de Confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)

#Calculo la precisión del modelo
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision)