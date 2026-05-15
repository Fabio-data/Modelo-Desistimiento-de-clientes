# Modelo de Desistimiento de Crédito con LightGBM

Modelo de **Machine Learning end-to-end** que predice la probabilidad de que un
cliente **desista** de su solicitud de crédito, usando **LightGBM**. Incluye EDA,
ingeniería de variables, ajuste de hiperparámetros, optimización del umbral,
interpretabilidad (SHAP) y despliegue en Streamlit.

## Enlaces

- **App en vivo (Streamlit):** https://modelo-desistimiento-de-clientes-gvdkeqyvdkx8gyfinsabmx.streamlit.app/
- **Informe del análisis (Quarto):** https://fabio-data.github.io/Modelo-Desistimiento-de-clientes/

## Datos

> **Aviso:** `Data_bancaria.xlsx` contiene **datos ficticios** creados para fines
> educativos. No corresponde a clientes reales y puede usarse libremente.

Variables demográficas, financieras y contractuales (edad, ingresos, egresos,
plazo, tipo de contrato, historial, etc.). La variable objetivo `DESISTE` se
deriva del estado final de la solicitud (`Desistida` = 1).

## Enfoque

1. **EDA y limpieza** — nulos, estados finales válidos, definición del target.
2. **Ingeniería de variables** — capacidad de pago, ratio de endeudamiento,
   ratio solicitud/ingreso y estrés financiero.
3. **Modelado** — `Pipeline` + `ColumnTransformer` (imputación + One-Hot),
   `LightGBM` con `scale_pos_weight` para el desbalanceo y `RandomizedSearchCV`
   (5-fold) para hiperparámetros.
4. **Evaluación** — split estratificado train/val/test; el **umbral** se
   optimiza por F1 en **validación** y se evalúa en **test**.
5. **Interpretabilidad** — importancia de variables y **SHAP**.

## Resultados (conjunto de test)

| Métrica | Valor |
|---|---|
| Umbral operativo | **0.41** |
| Recall (clase *desiste*) | **0.74** |
| Precision (clase *desiste*) | 0.48 |
| F1 (clase *desiste*) | 0.58 |
| Accuracy | 0.61 |

El modelo prioriza el **recall**: detecta la mayoría de los clientes que
desisten (útil para retención), a costa de más falsos positivos. Es un punto de
partida honesto, no un modelo de producción.

## Ejecutar localmente

```bash
# 1) App de Streamlit
pip install -r requirements.txt
streamlit run app.py

# 2) Notebook de análisis / reentrenamiento
pip install -r requirements-dev.txt
jupyter notebook modelo_LightGBM.ipynb
```

## Estructura

```
app.py                          # App de Streamlit (consume el modelo)
modelo_LightGBM.ipynb           # Notebook end-to-end
modelo_desistimiento_lgbm.joblib# Artefacto: modelo + umbral + columnas
Data_bancaria.xlsx              # Dataset ficticio
docs/                           # Informe Quarto publicado (GitHub Pages)
tests/test_smoke.py             # Prueba de humo (carga + predicción)
requirements.txt                # Dependencias de la app
requirements-dev.txt            # + dependencias del notebook
```

## Modelo entrenado

El artefacto `modelo_desistimiento_lgbm.joblib` guarda el `modelo`, el
`best_threshold` y las `feature_cols`, lo que permite reutilizarlo sin
reentrenar e integrarlo en la app.
