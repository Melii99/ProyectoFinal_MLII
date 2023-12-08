#!/usr/bin/env python
# coding: utf-8

# Importar librerías

# DataFrames
import pandas as pd
# Plots
import matplotlib.pyplot as plt
# Seeds
import numpy as np
# Support Vector Classifier (SVC)
from sklearn.svm import SVC
# Importar objeto para escalmiento de datos
from sklearn.preprocessing import StandardScaler
# Búsqueda aleatoria de hyperparámetros k-fold cross-validation
from sklearn.model_selection import RandomizedSearchCV
# Obtener valores de una distribución uniforme
from scipy.stats import uniform, expon
# Métricas de evaluación
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
# Matriz de confusión
from sklearn.metrics import confusion_matrix
# Reporte de clasificación 
from sklearn.metrics import classification_report
# Plot Precision-Recall curve
from sklearn.metrics import PrecisionRecallDisplay
# submuestreo a los datos de entrenamiento
from imblearn.under_sampling import RandomUnderSampler


# # Exploración de datos

# Leer datos de entrenamiento en un dataframe 
train_data = pd.read_csv("C:\\Users\\user\\Dropbox\\MachineLearning_II\\FinalProjectData\\EstrogenReceptorStatus_Train.csv")

train_data.head(10)

# Verificar dimensionalidad de datos de entrenamiento
train_data.shape

# Leer datos de evaluación en un dataframe
test_data = pd.read_csv("C:\\Users\\user\\Dropbox\\MachineLearning_II\\FinalProjectData\\EstrogenReceptorStatus_Test.csv")

test_data.head(10)

# Verificar dimensionalidad de datos de evaluacion
test_data.shape

# Crear dataframe de todos los datos (concatenar filas)
data = pd.concat([train_data, test_data], axis=0)
# Agregar header de los casos
data = data.rename(columns={"Unnamed: 0": "case"})

data.head(10)

# Verificar dimensionalidad de los datos
data.shape

# Agregar etiquetas de clase al dataset completo

# Dataframes de las etiquetas
train_labels = pd.read_csv("C:\\Users\\user\\Dropbox\\MachineLearning_II\\FinalProjectData\\EstrogenReceptorStatus_Train_labels.txt", header=None)
test_labels = pd.read_csv("C:\\Users\\user\\Dropbox\\MachineLearning_II\\FinalProjectData\\EstrogenReceptorStatus_Test_labels.txt", header=None)

# Crear dataframe de todas las etiquetas (concatenar filas)
data_labels = pd.concat([train_labels, test_labels], axis=0)

# Verificar dimensionalidad de las etiquetas
data_labels.shape

# Agregar etiquetas al dataframe de todos los datos (concatenar columnas)
data = pd.concat([data, data_labels], axis=1)

# Verificar dimensionalidad de los datos con etiquetas
data.shape

# Agregar header 'class' a las etiquetas en 'data'
data = data.rename(columns={0: "clases"})

# Nombres de las columnas de data
data.columns

# Agrupar y contar ocurrencias de cada categoría de clase (0 y 1)
grouped_data = data.groupby('clases')['case'].count()
grouped_data.head

# Gráfica de barras de número de casos por categoría de clase (0 y 1)

# Plot
plt.bar(grouped_data.index, grouped_data.values)
plt.xlabel('Categorías de clase')
plt.ylabel('Casos')
plt.title('Número de casos por Categorías de clase')

# Personalizar las etiquetas del eje x
labels = ['Estrógeno Negativo', 'Estrógeno Positivo']
xticks = [0, 1]  # Ubicaciones correspondientes a las categorías
plt.xticks(xticks, labels)

plt.show()

# Encontramos que existe un desbalance de clases por lo que podemos recurrir a hacer un "undersampling" o un "oversampling" de los datos para obtener una muestra balanceada o usar algúnos hiperparámetros como class_weight 'balanced' dentro del clasificador.

# Calcular el rango de valores máximos y mínimos en cada columna
rangos_max = data.max()
rangos_min = data.min()

# Crear un DataFrame con los resultados
rangos_df = pd.DataFrame({'Max': rangos_max, 'Min': rangos_min})
rangos_df = rangos_df.iloc[1:-1]

print(rangos_df)

rangos_df.shape

# Grafica de valores máximos y mínimos de cada metabolito

# PLot
plt.plot(rangos_df.index, rangos_df['Max'], label='Valores Máximos')
plt.plot(rangos_df.index, rangos_df['Min'], label='Valores Mínimos')
plt.xlabel('Metabolitos')
plt.ylabel('Valores')
plt.title('Valores Máximos y Mínimos de cada Metabolito')
plt.legend()
plt.show()

# El rango de valores para todos los metabolitos no es muy grande (los valores son relativamente similares), por lo que podría no sernecesario un escalamiento de los datos.

# Clasificador SVM con escalamiento de datos #

# Preparación de datos

# X = sólo datos de características de ejemplos
# Eliminamos la primera columna de 'case' para quedarnos sólo con los valores de los metabolitos
X_train = train_data.iloc[:, 1:]
X_test = test_data.iloc[:, 1:]

# Y = categorías de clase para cada ejemplo
Y_train = train_labels
Y_test = test_labels

# Escalamiento de datos

# Crear un escalador y ajustarlo a los datos de entrenamiento
scaler = StandardScaler()
scaler.fit(X_train)

# Aplicar el escalado a los datos de entrenamiento y prueba
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del clasificador

# Inicializar una semilla
seed = 42
np.random.seed(seed)

# Creación del clasificador SVM
svm_model = SVC()

# Hiperparámetros que se desean ajustar
search_parameters = {'C': expon(scale=100), 'gamma': expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}

# Búsqueda de hiperparámetros con RandomizedSearchCV  
clf = RandomizedSearchCV(svm_model, search_parameters, random_state=0, cv=15, 
                                    scoring='f1_weighted', verbose=2, n_jobs=-1)

# Ajuste del modelo a los datos de entrenamiento escalados
classifier = clf.fit(X_train_scaled, Y_train)

# Resultados de entrenamiento 
print("Mejor F-1 weighted score:")
print(classifier.best_score_)

# Resultados de cross-validation (clasificador)
print("Mejores hiperparámetros:")
print(classifier.best_params_)

# Evaluación del clasificador

# Predicciones en los datos de prueba usando el clasificador entrenado
Y_pred = classifier.predict(X_test_scaled)

# Resultados de la evaluación
print("Resultados de la evaluación:")
print("F-1 weighted: {}".format(f1_score(Y_test, Y_pred, average='weighted')))
print("Precision: {}".format(precision_score(Y_test, Y_pred)))
print("Recall: {}".format(f1_score(Y_test, Y_pred)))
print("AUROC: {}".format(roc_auc_score(Y_test, Y_pred, average='weighted')))

# Reporte de la clasificación de los datos de evaluación
print(classification_report(Y_test,Y_pred))

# Matriz de confusión
confusion_matrix = confusion_matrix(Y_test,Y_pred)
print("Matriz de confusión:")
print(confusion_matrix)

# Curva ROC

# Tasas de falsos positivos, verdaderos positivos y umbrales
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
# Calcular el área bajo la curva ROC (AUC-ROC) ponderada
auc_roc = roc_auc_score(Y_test, Y_pred, average='weighted')

# Plot
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % auc_roc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Clasificador SVM sin escalamiento de datos #

# Entrenamiento del clasificador

# Inicializar una semilla
seed = 42
np.random.seed(seed)

# Creación del clasificador SVM
svm_model1 = SVC()

# Hiperparámetros que se desean ajustar
search_parameters = {'C': expon(scale=100), 'gamma': expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}

# Búsqueda de hiperparámetros con RandomizedSearchCV  
clf1 = RandomizedSearchCV(svm_model1, search_parameters, random_state=0, cv=15, 
                                    scoring='f1_weighted', verbose=2, n_jobs=-1)

# Ajuste del modelo a los datos de entrenamiento sin escalar
classifier1 = clf1.fit(X_train, Y_train)

# Resultados de entrenamiento 
print("Mejor F-1 weighted score:")
print(classifier1.best_score_)

# Resultados de cross-validation (clasificador)
print("Mejores hiperparámetros:")
print(classifier1.best_params_)

# Evaluación del clasificador

# Predicciones en los datos de prueba usando el clasificador entrenado
Y_pred1 = classifier1.predict(X_test)

# Resultados de la evaluación
print("Resultados de la evaluación:")
print("F-1 weighted: {}".format(f1_score(Y_test, Y_pred1, average='weighted')))
print("Precision: {}".format(precision_score(Y_test, Y_pred1)))
print("Recall: {}".format(f1_score(Y_test, Y_pred1)))
print("AUROC: {}".format(roc_auc_score(Y_test, Y_pred1, average='weighted')))

# Reporte de la clasificación de los datos de evaluación
print(classification_report(Y_test,Y_pred1))

# Matriz de confusión

from sklearn.metrics import confusion_matrix

confusion_matrix1 = confusion_matrix(Y_test,Y_pred1)
print("Matriz de confusión:")
print(confusion_matrix1)

# Curva ROC

# Tasas de falsos positivos, verdaderos positivos y umbrales
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred1)
# Calcular el área bajo la curva ROC (AUC-ROC) ponderada
auc_roc1 = roc_auc_score(Y_test, Y_pred1, average='weighted')

# Plot
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % auc_roc1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Undersampling #

# Submuestreo de los datos de entrenamiento escalados
rus = RandomUnderSampler(random_state=0)

X_train_resampled, Y_train_resampled = rus.fit_resample(X_train_scaled, Y_train)

## Entrenamiento del clasificador

# Inicializar una semilla
seed = 42
np.random.seed(seed)

# Creación del clasificador SVM
svm_model_us = SVC()

# Hiperparámetros que se desean ajustar
search_parameters = {'C': expon(scale=100), 'gamma': expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':[None]}

# Búsqueda de hiperparámetros con RandomizedSearchCV  
clf_us = RandomizedSearchCV(svm_model_us, search_parameters, random_state=0, cv=15, 
                                    scoring='f1_weighted', verbose=2, n_jobs=-1)

# Ajuste del modelo a los datos de entrenamiento escalados
classifier_us = clf_us.fit(X_train_resampled, Y_train_resampled)

# Resultados de entrenamiento 
print("Mejor F-1 weighted score:")
print(classifier_us.best_score_)

# Resultados de cross-validation (clasificador)
print("Mejores hiperparámetros:")
print(classifier_us.best_params_)

# Evaluación del clasificador

# Predicciones en los datos de prueba usando el clasificador entrenado
Y_pred_us = classifier_us.predict(X_test_scaled)

# Resultados de la evaluación
print("Resultados de la evaluación:")
print("F-1 weighted: {}".format(f1_score(Y_test, Y_pred_us, average='weighted')))
print("Precision: {}".format(precision_score(Y_test, Y_pred_us)))
print("Recall: {}".format(f1_score(Y_test, Y_pred_us)))
print("AUROC: {}".format(roc_auc_score(Y_test, Y_pred_us, average='weighted')))

# Reporte de la clasificación de los datos de evaluación
print(classification_report(Y_test,Y_pred_us))

# Matriz de confusión

from sklearn.metrics import confusion_matrix

confusion_matrix_us = confusion_matrix(Y_test,Y_pred_us)
print("Matriz de confusión:")
print(confusion_matrix_us)

# Curva ROC

# Tasas de falsos positivos, verdaderos positivos y umbrales
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_us)
# Calcular el área bajo la curva ROC (AUC-ROC) ponderada
auc_roc_us = roc_auc_score(Y_test, Y_pred_us, average='weighted')

# Plot
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % auc_roc_us)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
