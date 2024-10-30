
"Análisis Exploratorio de Datos (EDA)"

"Actividad realizada de manera intividual"

"Importar bibliotecas:"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"Cargar el dataset que deseamos implementar teneindo encuenta la ruta del documento :"
df = pd.read_csv('ruta/al/dataset.csv')


"Exploramos  el dataset:"

df.info()      # Información general
df.describe()  # Estadísticas descriptivas
sns.pairplot(df)  # Gráficos de dispersión entre variables


"Identificar valores atípicos:"
sns.boxplot(data=df)

"-------------------------------------------------------"
"2. Preprocesamiento de Datos"
"Manejo de valores faltantes"

df.fillna(df.mean(), inplace=True)  # O cualquier otro método

"Transformaciones necesarias:"
df['columna'] = df['columna'].astype('categorical')  # Ejemplo de conversión

"--------------------------------------------------------"

"3. Selección de Características"
"Utilizar técnicas como correlación:"
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)

"Seleccionar características relevantes:"
from sklearn.feature_selection import SelectKBest, f_classif
X_new = SelectKBest(score_func=f_classif, k=10).fit_transform(X, y)

"--------------------------------------------------------"

"4. Dividir el Dataset"
"Dividir en conjuntos de entrenamiento y prueba"

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"-------------------------------------------------------"

"5. Entrenamiento del Modelo"
"Regresión Lineal:"

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

"Regresión Logística:"
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

"Árboles de Decisión:"
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

"--------------------------------------------------------"

"6. Evaluación del Modelo"
"Evaluar el desempeño:"

from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

"Para regresión:"
mse = mean_squared_error(y_test, predictions)
print(f'MSE: {mse}')

"--------------------------------------------------------"

"7. Visualización de Resultados"
"Gráficas de dispersión y predicciones"

plt.scatter(y_test, predictions)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs Predicciones')
plt.show()


