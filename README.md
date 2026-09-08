# Workshop 2 — Machine Learning (Unidad 2)
### Pipelines, entrenamiento, comparación de modelos y validación cruzada

Repositorio del equipo para el Workshop 2 de la materia de Machine Learning. Se resuelven dos problemas completos de ML supervisado — un problema de **regresión** y uno de **clasificación** — siguiendo el mismo ciclo metodológico: EDA y limpieza, pipelines con `ColumnTransformer`, entrenamiento y comparación de varios algoritmos, y validación cruzada.

## Integrantes

1. Alexander Vargas
2. Cristian Cabarcas López
3. Jacobo Pava Quintero
4. Samuel Herrera Galvis

## Estructura del repositorio

```
├── 1_Regresion_Vuelos_Workshop2_CORREGIDO.ipynb        # Problema de regresión
├── 2_Clasificacion_Tiroides_Workshop2_CORREGIDO.ipynb  # Problema de clasificación
├── Clean_Dataset_Flight_Price_Prediction.csv           # Dataset de vuelos (respaldo local)
├── Thyroid_Disease_Data.csv                            # Dataset de tiroides (respaldo local)
└── README.md
```

## Problema 1 — Regresión: Predicción de precios de vuelos

- **Dataset:** [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (Kaggle), 300,153 registros de vuelos domésticos en India (plataforma Ease My Trip).
- **Variable objetivo:** `price` (precio del tiquete, en rupias).
- **Modelos entrenados:** Regresión Lineal, KNN Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor (`HistGradientBoostingRegressor`).
- **Mejor modelo:** Random Forest — R² en Test = 0.9848, MAE en Test ≈ 1,103 rupias (~5.3% del precio promedio).
- El notebook carga el dataset primero desde Kaggle con `kagglehub`; si no hay conexión, usa automáticamente el archivo local `Clean_Dataset_Flight_Price_Prediction.csv`.

## Problema 2 — Clasificación: Diagnóstico de recurrencia de cáncer de tiroides

- **Dataset:** [Thyroid Disease Data](https://www.kaggle.com/datasets/jainaru/thyroid-disease-data/data) (Kaggle), 383 registros clínicos (364 tras eliminar duplicados).
- **Variable objetivo:** `Recurred` (Yes/No — recurrencia de la enfermedad tras el tratamiento inicial).
- **Modelos entrenados:** KNN Classifier, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier.
- **Mejor modelo:** Random Forest — Accuracy/F1 en Test = 0.964 / 0.938.
- El notebook carga el dataset primero desde Kaggle con `kagglehub`; si no hay conexión, usa automáticamente el archivo local `Thyroid_Disease_Data.csv`.

## Metodología común a ambos notebooks

1. **EDA y limpieza:** diccionario de datos, `.describe()`, nulos, duplicados, inconsistencias de formato, detección de outliers (criterio IQR) y gráficas de EDA (barras, pie, histograma, scatter, boxplot), cada una con interpretación y recomendación de negocio/clínica.
2. **Preprocesamiento:** encoding justificado por variable (One-Hot para nominales, Ordinal para variables con jerarquía natural) y escalamiento con `RobustScaler` para las variables numéricas, dada la presencia de outliers.
3. **División de datos:** Train / Validation / Test (70/15/15), con `stratify` sobre el target en el problema de clasificación.
4. **Pipelines:** `ColumnTransformer` + modelo dentro de un único `Pipeline` de scikit-learn, con verificación explícita de que el `.fit()` de todo el preprocesamiento se realiza únicamente sobre `X_train`.
5. **Evaluación:** métricas en Train y Validación (para detectar over/underfitting), evaluación final única sobre Test, y **5-Fold Cross Validation** sobre el mejor modelo de cada problema.
6. **Conclusiones (Fase 8):** comparación conjunta de ambos problemas — qué modelo generalizó mejor y por qué, qué limpieza fue más determinante, limitaciones del estudio y preguntas abiertas — acompañada de un subplot (2×2) por notebook con las gráficas más relevantes.

## Cómo ejecutar

1. Clonar el repositorio.
2. Abrir cada notebook en Jupyter o Google Colab.
3. Ejecutar las celdas en orden. Cada notebook intenta descargar su dataset desde Kaggle con `kagglehub`; si esto falla (sin conexión o sin credenciales de Kaggle configuradas), se usa automáticamente el archivo `.csv` correspondiente incluido en este repositorio, siempre que esté en el mismo directorio que el notebook.

## Sustentación

Este workshop puede ser sustentado por integrantes seleccionados aleatoriamente en representación del equipo. La nota final se calcula como `(Sustentación + Entrega) / 2`.
