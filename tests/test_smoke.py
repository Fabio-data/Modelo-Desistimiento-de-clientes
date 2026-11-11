"""Pruebas rápidas: no dependen del Excel ni de internet, solo de lo versionado."""
import json

import joblib
import pandas as pd
import pytest

from features import DERIVADAS, agregar_variables, preparar_datos

ARTIFACT = "modelo_desistimiento_lgbm.joblib"


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(ARTIFACT)


def test_variables_derivadas():
    df = agregar_variables(pd.DataFrame(
        [{"INGRESOS": 2_000_000.0, "EGRESOS": 500_000.0, "VALOR_SOLICITADO": 4_000_000.0}]))
    fila = df.iloc[0]
    assert fila["CAPACIDAD_PAGO"] == 1_500_000
    assert fila["RATIO_ENDEUDAMIENTO"] == pytest.approx(0.25)
    assert fila["RATIO_SOLICITUD_INGRESO"] == pytest.approx(2.0)
    assert fila["ESTRES_FINANCIERO"] == pytest.approx(2.25)


def test_preparar_datos_limpia_y_define_target():
    crudo = pd.DataFrame({
        "SOLICITUD": [1, 2, 3], "FECHA_INICIO": ["2025-01-01"] * 3,
        "GENERO": ["F", "M", "F"], "Marca producto": ["Marca A"] * 3,
        "Estado": ["Desistida", "Aprobada", "En estudio"],
        "TIPO_CONTRATO": [None, "Indefinido", "Indefinido"],
        "EDAD": [2000, 30, 40],
        "INGRESOS": [1.0, 2.0, 3.0], "EGRESOS": [1.0, 1.0, 1.0],
        "VALOR_SOLICITADO": [5.0, 5.0, 5.0],
    })
    X, y = preparar_datos(crudo)
    assert y.tolist() == [1, 0]  # "En estudio" no es un estado final
    assert not {"GENERO", "Estado", "SOLICITUD"} & set(X.columns)
    assert X["TIPO_CONTRATO"].tolist() == ["Otra", "Indefinido"]
    assert pd.isna(X["EDAD"].iloc[0])  # edad imposible -> nulo


def test_artifact_predice_con_valores_tipicos(artifact):
    assert 0.0 < artifact["best_threshold"] < 1.0
    # Mismo camino que la app: valores típicos + variables derivadas.
    fila = agregar_variables(pd.DataFrame([artifact["valores_tipicos"]]))
    assert set(artifact["feature_cols"]) == set(fila.columns)
    assert set(DERIVADAS) <= set(fila.columns)

    proba = artifact["modelo"].predict_proba(fila[artifact["feature_cols"]])[0, 1]
    assert 0.0 <= proba <= 1.0


def test_modelo_supera_a_los_baselines():
    with open("metrics.json", encoding="utf-8") as fh:
        metricas = {m["modelo"]: m for m in json.load(fh)["comparacion_test"]}
    final = metricas["LightGBM calibrado"]
    assert final["pr_auc"] > metricas["Reg. Logística"]["pr_auc"]
    assert final["roc_auc"] > metricas["Reg. Logística"]["roc_auc"] > 0.5
