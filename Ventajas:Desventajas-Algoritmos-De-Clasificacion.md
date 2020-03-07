# Ventajas y Desventajas de: Algoritmos de Clasificación

Material obtenido de:  [ligdigonzalez.com](hhttps://ligdigonzalez.com)

![Texto alternativo](algoritmos_de_clasificacion.png)

## Regresión Logística

#### Ventajas:

* Fácil de entender y explicar
* Rara vez existe sobreajuste
* El uso de la regularización es efectivo en la selección de funciones.
* Rápido para entrenar.
* Fácil de entrenar sobre grandes datos gracias a su versión estocástica.

#### Desventajas:

* Tienes que trabajar duro para que se ajuste a los datos no lineales.
* Puede sufrir con valores atípicos.
* En algunas ocasiones es muy simple para captar relaciones complejas entre variables.

#### Posibles usos:

* Ordenar los resultados por probabilidad
* Modelado de respuestas de marketing


## KNN (K Vecinos más Cercanos)

#### Ventajas:

* Simple
* Potente
* Entrenamiento rápido
* Puede manejar naturalmente problemas extremos de multiclases, como etiquetado de texto.

#### Desventajas:

* Costoso y lento para predecir nuevas instancias.
* Se debe definir una función de distancia significativa.
* Funciona mal en conjuntos de datos de alta dimensionalidad.

#### Posibles usos:

* Conjuntos de datos de baja dimensión
* Visión por computador
* Seguridad informática: detección de intrusos
* Detección de fallos en la fabricación de semiconductores
* Sistema de recomendación
* Problemas de corrección ortográfica
* Recuperación de contenido de video


## SVC (Máquines Vectores de Soporte Clasificación)

#### Ventajas:

* Se pueden modelar relaciones complejas, no lineales.
* Robusto al ruido, esto se debe a que maximizan los márgenes.

#### Desventajas:

* Necesidad de seleccionar una buena función de kernel.
* Los parámetros del modelo son difíciles de interpretar.
* Requiere memoria significativa y poder de procesamiento.
* Cuando se tiene muchos datos toma demasiado tiempo para entrenar.

#### Posibles usos:

* Clasificación de texto e imágenes.
* Reconocimiento de escritura a mano.


## Naive Bayes

#### Ventajas: 

* Fácil y rápido de implementar.
* No requiere demasiada memoria y se puede utilizar para el aprendizaje en línea.
* Fácil de entender.

#### Desventajas:

* Falla al estimar las características raras.
* Sufre al tener características irrelevantes.

#### Posibles usos:

* Reconocimiento de rostros
* Análisis de los sentimientos
* Detección de spam
* Clasificación de textos


## Árboles de Decisión Clasificación

#### Ventajas: 

* Muy fácil de interpretar y entender.
* Rápido.
* Robusto al ruido y valores perdidos.
* Preciso
* Excelente para aprender relaciones complejas, altamente no lineales. Por lo general, pueden lograr un rendimiento bastante alto.

#### Desventajas:

* Los árboles complejos son difíciles de interpretar.
* Es posible la duplicación dentro del mismo subárbol.
* En ocasiones no es utilizado por ser un algoritmo tan sencillo y no tan poderoso para datos complejos.

#### Posibles usos:

* Diagnóstico médico.
* Análisis de riesgo crediticio.


## Bosques Aleatorios Clasificación

#### Ventajas: 

* Puede trabajar en paralelo.
* Rara vez se sobreajusta.
* Maneja automáticamente los valores perdidos.
* No es necesario transformar ninguna variable.
* No hay necesidad de ajustar parámetros.
* Puede ser utilizado por casi cualquier persona con excelentes resultados.

#### Desventajas:

* Difícil de interpretar.
* Parcialmente en problemas multiclase hacia clases más frecuentes.

#### Posibles usos:

* Para casi cualquier problema de Machine Learning.
* Bioinformática.