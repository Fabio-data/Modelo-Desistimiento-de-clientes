"""Prueba de humo: el artefacto carga y produce una probabilidad válida.

No depende del Excel ni de internet: usa solo el .joblib versionado.
"""
import numpy as np
import pandas as pd
import joblib

ARTIFACT = "modelo_desistimiento_lgbm.joblib"


def test_artifact_estructura():
    art = joblib.load(ARTIFACT)
    assert {"modelo", "best_threshold", "feature_cols"} <= set(art)
    assert 0.0 < float(art["best_threshold"]) < 1.0
    assert len(art["feature_cols"]) > 0


def test_prediccion_valida():
    art = joblib.load(ARTIFACT)
    modelo, feature_cols = art["modelo"], art["feature_cols"]

    # Fila mínima: numéricas con un valor, el pipeline imputa el resto.
    fila = pd.DataFrame([{c: np.nan for c in feature_cols}])
    for c in ("INGRESOS", "EGRESOS", "VALOR_SOLICITADO"):
        if c in fila.columns:
            fila[c] = 2_000_000.0

    proba = float(modelo.predict_proba(fila)[0, 1])
    assert 0.0 <= proba <= 1.0
