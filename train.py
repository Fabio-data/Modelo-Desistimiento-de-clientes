"""Entrenamiento reproducible del modelo de desistimiento.

Flujo:
1. Split estratificado train / validación / test.
2. Baselines (Dummy y Regresión Logística) para tener una referencia.
3. LightGBM con búsqueda de hiperparámetros por PR-AUC (datos desbalanceados).
4. Calibración isotónica con validación cruzada sobre train.
5. Umbral operativo elegido por F1 en validación (datos que la calibración no vio).
6. Evaluación única en test y guardado del artefacto + metrics.json.

Uso:  python train.py
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import DERIVADAS, preparar_datos

DATA_PATH = "Data_bancaria.xlsx"
ARTIFACT_PATH = "modelo_desistimiento_lgbm.joblib"
METRICS_PATH = "metrics.json"
RANDOM_STATE = 42

ESPACIO_BUSQUEDA = {
    "clf__n_estimators": [300, 500, 800],
    "clf__learning_rate": [0.01, 0.03, 0.1],
    "clf__num_leaves": [20, 31, 50],
    "clf__max_depth": [-1, 7, 10],
    "clf__subsample": [0.7, 0.9, 1.0],
    "clf__colsample_bytree": [0.7, 0.9, 1.0],
    "clf__min_child_samples": [20, 50, 100],
    "clf__reg_lambda": [0, 1, 5],
}


def construir_pipeline(clf, escalar: bool = False) -> Pipeline:
    """Preprocesado (imputación + one-hot) seguido del clasificador."""
    pasos_num = [("imputer", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("scaler", StandardScaler()))
    categoricas = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", Pipeline(pasos_num), make_column_selector(dtype_include="number")),
        ("cat", categoricas, make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluar(nombre: str, y_true, proba, umbral: float = 0.5) -> dict:
    pred = (proba >= umbral).astype(int)
    return {
        "modelo": nombre,
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "f1_desiste": round(f1_score(y_true, pred, zero_division=0), 4),
        "recall_desiste": round(recall_score(y_true, pred, zero_division=0), 4),
        "precision_desiste": round(precision_score(y_true, pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "threshold": round(float(umbral), 4),
    }


def umbral_optimo_f1(y_true, proba) -> float:
    precision, recall, umbrales = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(umbrales[np.argmax(f1[:-1])])


def main() -> None:
    X, y = preparar_datos(pd.read_excel(DATA_PATH))
    print(f"Datos: {len(X)} filas, {X.shape[1]} variables | desisten: {y.mean():.1%}")

    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto, y_resto, test_size=0.2, random_state=RANDOM_STATE, stratify=y_resto)

    def proba_test(modelo):
        return modelo.fit(X_train, y_train).predict_proba(X_test)[:, 1]

    # ---- Baselines ----
    dummy = construir_pipeline(DummyClassifier(strategy="prior"))
    logreg = construir_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"), escalar=True)
    comparacion = [
        evaluar("Dummy (baseline)", y_test, proba_test(dummy)),
        evaluar("Reg. Logística", y_test, proba_test(logreg)),
    ]

    # ---- LightGBM + búsqueda de hiperparámetros por PR-AUC ----
    lgbm = lgb.LGBMClassifier(
        objective="binary", random_state=RANDOM_STATE, verbose=-1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    busqueda = RandomizedSearchCV(
        construir_pipeline(lgbm), ESPACIO_BUSQUEDA, n_iter=20, cv=4,
        scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1)
    busqueda.fit(X_train, y_train)
    mejor = busqueda.best_estimator_
    print("Mejores parámetros:", busqueda.best_params_)
    comparacion.append(evaluar(
        "LightGBM (sin calibrar)", y_test, mejor.predict_proba(X_test)[:, 1]))

    # ---- Calibración (CV sobre train) y umbral (validación) ----
    calibrado = CalibratedClassifierCV(mejor, method="isotonic", cv=5, ensemble=False)
    calibrado.fit(X_train, y_train)
    umbral = umbral_optimo_f1(y_val, calibrado.predict_proba(X_val)[:, 1])

    # ---- Evaluación final en test ----
    final = evaluar(
        "LightGBM calibrado", y_test, calibrado.predict_proba(X_test)[:, 1], umbral)
    comparacion.append(final)

    print(f"\n{'modelo':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>8}{'umbral':>8}")
    for r in comparacion:
        print(f"{r['modelo']:<26}{r['roc_auc']:>9}{r['pr_auc']:>9}"
              f"{r['f1_desiste']:>8}{r['threshold']:>8}")

    # ---- Artefacto: modelo + umbral + metadatos que usa la app ----
    base = X_train.drop(columns=DERIVADAS)
    numericas = base.select_dtypes(include="number").columns
    categoricas = base.columns.difference(numericas)
    joblib.dump(
        {
            "modelo": calibrado,
            "best_threshold": umbral,
            "feature_cols": X.columns.tolist(),
            # Valor típico por variable: prellena el formulario de la app.
            "valores_tipicos": {
                **base[numericas].median().to_dict(),
                **base[categoricas].mode().iloc[0].to_dict(),
            },
            "categorias": {c: sorted(base[c].dropna().unique()) for c in categoricas},
        },
        ARTIFACT_PATH,
        compress=3,
    )
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"comparacion_test": comparacion, "final": final}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\nGuardado: {ARTIFACT_PATH} y {METRICS_PATH}")


if __name__ == "__main__":
    main()
