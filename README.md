# Modelo de Desistimiento de Crédito con LightGBM

Este proyecto desarrolla un **modelo de Machine Learning** para predecir el **desistimiento de clientes** en un contexto crediticio, utilizando **LightGBM**.  
El enfoque es completamente **end-to-end**, desde la exploración de los datos hasta la evaluación e interpretación del modelo.

## Demo en vivo

**Prueba el modelo aquí:**  
https://modelo-desistimiento-de-clientes-gvdkeqyvdkx8gyfinsabmx.streamlit.app/

## ¿Cómo funciona?

1. El usuario ingresa información básica del cliente.
2. El modelo calcula la **probabilidad de desistimiento**.
3. Si la probabilidad supera un **umbral operativo**, el cliente se marca como **riesgo potencial**.


---

## Objetivo

Construir un modelo predictivo capaz de **estimar la probabilidad de que un cliente desista**, con el fin de:

- Identificar clientes con alto riesgo de desistimiento  
- Analizar las variables más influyentes en la decisión  
- Apoyar estrategias de retención y gestión de riesgo  

---

## Dataset

El conjunto de datos contiene información **demográfica, financiera y contractual** de los clientes.

Algunas variables incluidas:
- Edad
- Género
- Estado civil
- Multas / historial financiero
- Cuota inicial
- Plazo
- Variables relacionadas con el crédito

El dataset se encuentra en el archivo: Data_bancaria.xlsx 


---

## Enfoque del modelo

El pipeline de modelado incluye:

1. **Análisis exploratorio de datos (EDA)**
   - Distribución de variables
   - Identificación de valores faltantes
   - Análisis inicial de correlaciones

2. **Preparación de datos**
   - Separación de variables numéricas y categóricas
   - Imputación de valores faltantes
   - Codificación de variables categóricas
   - División train / test

3. **Modelado**
   - Modelo principal: **LightGBM Classifier**
   - Uso de `Pipeline` y `ColumnTransformer`
   - Búsqueda de hiperparámetros con `RandomizedSearchCV`

4. **Evaluación**
   - Matriz de confusión
   - Precision, Recall y F1-score
   - Curva Precision-Recall

5. **Interpretabilidad**
   - Importancia de variables
   - Análisis del impacto de los principales features
   - Interpretación orientada a negocio

---

## Tecnologías utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- LightGBM  
- SHAP  
- Matplotlib  
- Seaborn  

---

## Resultados

- El modelo LightGBM presenta un **buen desempeño predictivo** para identificar clientes con riesgo de desistimiento.
- Aunque la data esta desbalanceada, le dimos mas peso a la variable(desiste) para que logre obtener la mayoria de clientes.
- Se identifican **variables clave** con alto impacto en la predicción.
- La interpretabilidad del modelo permite **explicar las decisiones**, facilitando su adopción en contextos reales de negocio.

---

## Modelo entrenado
 
El modelo final entrenado se guarda en el archivo: modelo_desistimiento_lgbm.joblib 


Esto permite:
- Reutilizar el modelo sin reentrenar
- Integrarlo en aplicaciones o servicios posteriores

---
