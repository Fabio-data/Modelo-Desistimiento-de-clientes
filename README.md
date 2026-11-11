# Modelo de Desistimiento de Crédito con LightGBM

[![CI](https://github.com/Fabio-data/Modelo-Desistimiento-de-clientes/actions/workflows/ci.yml/badge.svg)](https://github.com/Fabio-data/Modelo-Desistimiento-de-clientes/actions/workflows/ci.yml)

Predice la probabilidad de que un cliente **desista** de su solicitud de crédito,
para que el equipo comercial sepa a quién contactar antes de perderlo. Proyecto
de punta a punta: datos, entrenamiento reproducible, pruebas y app desplegada.

- **App en vivo (Streamlit):** https://modelo-desistimiento-de-clientes-gvdkeqyvdkx8gyfinsabmx.streamlit.app/
- **Informe del análisis (Quarto):** https://fabio-data.github.io/Modelo-Desistimiento-de-clientes/

## Resultado en una línea

Con un umbral de 0.30 el modelo detecta **7 de cada 10 clientes que desisten**
(recall 0.72), con una precisión de 0.49: de cada dos alertas, una es real.

| Modelo (conjunto de test) | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Dummy (clase mayoritaria) | 0.50 | 0.36 | 0.00 |
| Regresión Logística | 0.63 | 0.48 | 0.52 |
| LightGBM (sin calibrar) | 0.72 | 0.62 | 0.57 |
| **LightGBM calibrado** | **0.72** | **0.62** | **0.58** |

**Lectura honesta:** la capacidad discriminativa es moderada (ROC-AUC 0.72) y el
dataset limita el techo. El valor frente a los baselines es claro (PR-AUC de 0.48
a 0.62) y las probabilidades están calibradas, así que el "%" que muestra la app
se puede interpretar como tal. Es un punto de partida sólido y reproducible, no
un modelo de producción. Métricas completas en [`metrics.json`](metrics.json).

## Datos

`Data_bancaria.xlsx` contiene **datos ficticios** creados con fines educativos:
33.668 solicitudes con variables demográficas, financieras y contractuales
(edad, ingresos, egresos, plazo, tipo de contrato, puntaje de riesgo, etc.).
El 36 % desiste. La variable objetivo `DESISTE` vale 1 cuando el estado final de
la solicitud es `Desistida`.

## Cómo está hecho

1. **Preparación** (`features.py`): se conservan solo solicitudes con estado
   final, se descartan identificadores y el género, se anulan edades imposibles
   y se crean cuatro variables de negocio (capacidad de pago, endeudamiento,
   solicitud/ingreso y estrés financiero). La app usa esta misma función, así
   que entrenamiento y predicción no pueden divergir.
2. **Modelado** (`train.py`): `Pipeline` con imputación y One-Hot, LightGBM con
   `scale_pos_weight` para el desbalanceo y `RandomizedSearchCV` optimizando
   PR-AUC.
3. **Calibración y umbral:** calibración isotónica con validación cruzada sobre
   train; el umbral se elige por F1 en validación y se evalúa una sola vez en
   test. Cada decisión usa datos que el paso anterior no vio.
4. **Interpretabilidad:** importancia de variables y SHAP en el informe Quarto.

## Ejecutar localmente

```bash
# App de Streamlit
pip install -r requirements.txt
streamlit run app.py

# Reentrenar (genera el .joblib y metrics.json) y probar
pip install -r requirements-dev.txt
python train.py
pytest -q
ruff check .
```

## Estructura

```
features.py                       # Limpieza y variables derivadas (train + app)
train.py                          # Entrenamiento reproducible -> .joblib + metrics.json
app.py                            # App de Streamlit
modelo_desistimiento_lgbm.joblib  # Modelo calibrado, umbral y metadatos para la app
metrics.json                      # Métricas y comparación contra baselines
tests/test_smoke.py               # Pruebas de datos, artefacto y métricas
modelo_LightGBM.ipynb             # Notebook de análisis exploratorio
docs/                             # Informe Quarto publicado (GitHub Pages)
Data_bancaria.xlsx                # Dataset ficticio
```

## Limitaciones

- Datos ficticios: los resultados no se trasladan a una cartera real.
- Validación con split aleatorio, no temporal; en producción habría que validar
  con solicitudes posteriores a las de entrenamiento.
- El informe Quarto documenta la exploración inicial; las métricas vigentes son
  las de `train.py` y `metrics.json`.

## Autor

Fabio Andrés Marulanda · [GitHub](https://github.com/Fabio-data)
