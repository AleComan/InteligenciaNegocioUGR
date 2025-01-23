# Prácticas de Inteligencia de Negocio

Este repositorio contiene las prácticas realizadas para la asignatura **Inteligencia de Negocio**. Cada práctica aborda un enfoque distinto dentro del análisis de datos, incluyendo clasificación, clustering y regresión. A continuación, se describe cada una de las prácticas en detalle.

## Tabla de Contenidos

- [1. Clasificación](#1.-clasificación)

- [2. Clustering](#2.-clustering)

- [3. Clustering](#3.-regresión)

- [Estructura del repositorio](#estructura-del-repositorio)

- [Recursos Adicionales](#recursos-adicionales)

- [Autor](#autor)

---

## 1. Clasificación

En esta práctica se realizaron análisis predictivos mediante clasificación utilizando la herramienta **KNIME**. Se trabajó con tres problemas distintos:

1. **Predicción de aprobación de créditos**  
   - Dataset: [Loan Approval Prediction](https://www.kaggle.com/datasets/itshappy/ps4e9-original-data-loan-approval-prediction/data)  
   - Datos: 32,581 instancias con 12 atributos (8 numéricos y 4 categóricos).  
   - Algoritmos utilizados: Random Forest, Árbol de Decisión, Naive Bayes, Gradient Boosted Trees, Stochastic Gradient Descent (SGD).  
   - Resultados: Se analizaron métricas como matrices de confusión, curvas ROC y medidas de desempeño ajustadas al desbalanceo de clases.

2. **Predicción de una segunda cita**  
   - Dataset: [Second Date Prediction](https://www.openml.org/search?type=data&sort=runs&id=40536&status=active)  
   - Datos: Información obtenida de eventos de citas rápidas con atributos como atractivo, sinceridad, inteligencia, entre otros.  
   - Algoritmos utilizados: Random Forest, Naive Bayes, XGBoost, Gradient Boosted Trees.  

3. **Predicción del tipo de enfermedad eritemato-escamosa**  
   - Dataset: [Dermatology Dataset](https://archive.ics.uci.edu/dataset/33/dermatology)  
   - Datos: Evaluaciones clínicas e histopatológicas con 34 atributos en total.  
   - Algoritmos utilizados: KNN, Support Vector Machine, Gradient Boosted Trees, Random Forest.  

---

## 2. Clustering

Esta práctica abordó técnicas de agrupamiento para segmentación de datos en el contexto de **alojamientos turísticos en Granada**. Se utilizó **Python** con librerías como `numpy`, `pandas`, `seaborn` y `scikit-learn`.

### Casos de estudio:
En todos los casos de estudio hemos utilizado los mismos algoritmos: K-Means, MeanShift, DBSCAN, BIRCH, Jerárquico Aglomerativo de Enlace Simple.  
1. **Alojamientos cercanos al centro de la ciudad**  
   - Variables analizadas: Precio, calidad, puntuación, tipo de alojamiento.  

2. **Hoteles de cuatro y cinco estrellas**  
   - Variables: Calidad, puntuación.

3. **Alojamientos baratos y competitivos**  
   - Análisis basado en precios bajos con respecto al precio medio de las consultas.

Se evaluaron métricas como el coeficiente de Silhouette, el índice de Calinski-Harabasz y el índice de Davies-Bouldin para determinar la calidad de los clusters.

---

## 3. Regresión

En esta práctica se participó en la competición [Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9) de Kaggle para predecir el precio de coches usados. Se emplearon diversos algoritmos y técnicas avanzadas:

- **Entorno:** Python en Jupyter Notebook con librerías como `numpy`, `pandas`, `scikit-learn`, y `H2O AutoML`.  
- **Algoritmos utilizados:** CatBoost, LightGBM, Random Forest, Redes Neuronales, XGBoost.  
- **Estrategias aplicadas:**  
  - Imputación de valores perdidos.  
  - Transformación de la variable objetivo a escala logarítmica.  
  - Combinación de modelos (ensemble y stacking).  

### Resultados
Se generaron múltiples experimentos con diferentes configuraciones, optimizando parámetros para mejorar el rendimiento según la métrica RMSE.

---

## Estructura del Repositorio

- `/Práctica 1. Clasificación/`: Flujos de trabajo de KNIME, datasets utilizados (CSV) y memoria de la práctica de clasificación.
- `/Práctica 2. Clustering/`: Notebooks de Jupyter, datasets utilizados (CSV) y memoria de la práctica de clustering.
- `/Práctica 3. Regresión/`: Notebooks de Jupyter, datasets utilizados (CSV) y memoria de la práctica de regresión.
---

## Recursos Adicionales

- KNIME: [https://www.knime.com](https://www.knime.com)  
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)  
- H2O AutoML: [https://docs.h2o.ai](https://docs.h2o.ai)

---

## Autor

**Alejandro Coman Venceslá**  
Estudiante de Ingeniería Informática y Administración y Dirección de Empresas, Universidad de Granada.
