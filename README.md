# Modelo de Desistimiento de Crédito con LightGBM

Modelo de **Machine Learning end-to-end** que predice la probabilidad de que un
cliente **desista** de su solicitud de crédito, usando **LightGBM**. Incluye EDA,
ingeniería de variables, ajuste de hiperparámetros, optimización del umbral,
interpretabilidad (SHAP) y despliegue en Streamlit.

## Enlaces

- **App en vivo (Streamlit):** https://modelo-desistimiento-de-clientes-gvdkeqyvdkx8gyfinsabmx.streamlit.app/
- **Informe del análisis (Quarto):** https://fabio-data.github.io/Modelo-Desistimiento-de-clientes/
- **Portafolio del autor:** https://fabio-marulanda-portfolio.vercel.app/

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

Comparación contra baselines (métricas independientes del umbral):

| Modelo | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Dummy (clase mayoritaria) | 0.50 | 0.36 | 0.00 |
| Regresión Logística | 0.63 | 0.48 | 0.52 |
| LightGBM (sin calibrar) | 0.72 | 0.62 | 0.57 |
| **LightGBM calibrado** | **0.72** | **0.61** | **0.58** |

Modelo final (LightGBM calibrado, umbral operativo **0.33**):

| Métrica (clase *desiste*) | Valor |
|---|---|
| Recall | **0.70** |
| Precision | 0.49 |
| F1 | 0.58 |
| Accuracy | 0.63 |
| ROC-AUC / PR-AUC | 0.72 / 0.61 |

**Lectura honesta:** la capacidad discriminativa es **moderada** (ROC-AUC ≈ 0.72)
y el dataset disponible limita el techo de desempeño. El valor real frente a un
baseline está claro: PR-AUC sube de 0.36 (azar) y 0.48 (regresión logística) a
0.61, y las **probabilidades están calibradas** (isotónica), por lo que el "%"
que muestra la app es interpretable. El modelo prioriza **recall** (detecta la
mayoría de los que desisten) a costa de falsos positivos. Es un punto de partida
sólido y reproducible (`python train.py`), no un modelo de producción.

> Métricas completas y comparación en `metrics.json`.

## Ejecutar localmente

```bash
# 1) App de Streamlit
pip install -r requirements.txt
streamlit run app.py

# 2) Reentrenar el modelo (reproducible) -> genera el .joblib y metrics.json
pip install -r requirements-dev.txt
python train.py

# 3) Notebook de análisis exploratorio
jupyter notebook modelo_LightGBM.ipynb
```

## Estructura

```
app.py                          # App de Streamlit (consume el modelo)
train.py                        # Entrenamiento reproducible -> .joblib + metrics
modelo_LightGBM.ipynb           # Notebook de análisis exploratorio
modelo_desistimiento_lgbm.joblib# Artefacto: modelo calibrado + umbral + columnas
metrics.json                    # Métricas y comparación vs baselines
Data_bancaria.xlsx              # Dataset ficticio
docs/                           # Informe Quarto publicado (GitHub Pages)
tests/test_smoke.py             # Prueba de humo (carga + predicción)
requirements.txt                # Dependencias de la app
requirements-dev.txt            # + dependencias de entrenamiento/notebook
```

## Modelo entrenado

El artefacto `modelo_desistimiento_lgbm.joblib` guarda el `modelo`, el
`best_threshold` y las `feature_cols`, lo que permite reutilizarlo sin
reentrenar e integrarlo en la app.
