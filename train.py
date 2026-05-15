"""
Entrenamiento reproducible del modelo de desistimiento.

Mejoras frente al notebook original:
- Baselines de referencia (Dummy y Regresión Logística).
- Búsqueda de hiperparámetros optimizada por PR-AUC (average_precision),
  más adecuada que F1 para datos desbalanceados.
- Calibración de probabilidades (isotónica) sobre validación.
- Umbral operativo elegido por F1 en validación.
- Métricas completas en test: ROC-AUC, PR-AUC, classification_report.

Genera el artefacto `modelo_desistimiento_lgbm.joblib`
({modelo, best_threshold, feature_cols}) compatible con app.py.

Uso:  python train.py
"""
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, average_precision_score, f1_score,
)
import lightgbm as lgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Carga y preparación (idéntica al notebook -> mismas feature_cols)
# ----------------------------------------------------------------------
def cargar_datos(path="Data_bancaria.xlsx"):
    df = pd.read_excel(path)
    df["TIPO_CONTRATO"] = df["TIPO_CONTRATO"].fillna("Otra")

    estados_finales = ["Desistida", "Negada", "Aprobada", "Anulada"]
    df = df[df["Estado"].isin(estados_finales)].copy()

    df.drop(columns=["SOLICITUD", "FECHA_INICIO", "GENERO", "Marca producto"],
            inplace=True)

    df["DESISTE"] = (df["Estado"] == "Desistida").astype(int)
    df = df.drop(columns=["Estado"])

    # Ingeniería de variables de negocio
    df["CAPACIDAD_PAGO"] = df["INGRESOS"] - df["EGRESOS"]
    df["RATIO_ENDEUDAMIENTO"] = df["EGRESOS"] / (df["INGRESOS"] + 1e-6)
    df["RATIO_SOLICITUD_INGRESO"] = df["VALOR_SOLICITADO"] / (df["INGRESOS"] + 1e-6)
    df["ESTRES_FINANCIERO"] = df["RATIO_ENDEUDAMIENTO"] + df["RATIO_SOLICITUD_INGRESO"]

    X = df.drop(columns=["DESISTE"])
    y = df["DESISTE"]
    return X, y


def build_preprocessor():
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num, make_column_selector(dtype_exclude="object")),
        ("cat", cat, make_column_selector(dtype_include="object")),
    ])


def evaluar(nombre, y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "modelo": nombre,
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "f1_desiste": round(f1_score(y_true, pred), 4),
        "threshold": round(float(thr), 4),
    }


def main():
    X, y = cargar_datos()
    print(f"Datos: {X.shape[0]} filas, {X.shape[1]} variables | "
          f"desisten: {y.mean()*100:.1f}%")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp)

    pre = build_preprocessor()
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    resumen = []

    # ---- Baseline 1: Dummy (clase mayoritaria) ----
    dummy = Pipeline([("pre", pre), ("clf", DummyClassifier(strategy="prior"))])
    dummy.fit(X_train, y_train)
    p = dummy.predict_proba(X_test)[:, 1]
    resumen.append(evaluar("Dummy (baseline)", y_test, p, 0.5))

    # ---- Baseline 2: Regresión Logística ----
    logreg = Pipeline([
        ("pre", pre),
        ("sc", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    logreg.fit(X_train, y_train)
    p = logreg.predict_proba(X_test)[:, 1]
    resumen.append(evaluar("Reg. Logística", y_test, p, 0.5))

    # ---- Modelo principal: LightGBM + búsqueda por PR-AUC ----
    lgbm = lgb.LGBMClassifier(objective="binary", random_state=RANDOM_STATE,
                              n_jobs=-1, verbose=-1, scale_pos_weight=spw)
    pipe = Pipeline([("pre", pre), ("clf", lgbm)])
    space = {
        "clf__n_estimators": [300, 500, 800],
        "clf__learning_rate": [0.01, 0.03, 0.1],
        "clf__num_leaves": [20, 31, 50],
        "clf__max_depth": [-1, 7, 10],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.9, 1.0],
        "clf__min_child_samples": [20, 50, 100],
        "clf__reg_lambda": [0, 1, 5],
    }
    print("Buscando hiperparámetros (scoring = average_precision / PR-AUC)...")
    search = RandomizedSearchCV(
        pipe, space, n_iter=20, cv=4, scoring="average_precision",
        random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("Mejores parámetros:", search.best_params_)

    p = best.predict_proba(X_test)[:, 1]
    resumen.append(evaluar("LightGBM (sin calibrar)", y_test, p, 0.5))

    # ---- Calibración de probabilidades (isotónica, sobre validación) ----
    calibrado = CalibratedClassifierCV(best, method="isotonic", cv="prefit")
    calibrado.fit(X_val, y_val)

    # ---- Umbral óptimo por F1 en validación (con prob. calibradas) ----
    proba_val = calibrado.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1s = np.nan_to_num(2 * prec * rec / (prec + rec))
    best_threshold = float(thr[np.argmax(f1s[:-1])])

    # ---- Evaluación final en test ----
    proba_test = calibrado.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= best_threshold).astype(int)
    resumen.append(evaluar("LightGBM calibrado", y_test, proba_test, best_threshold))

    print("\n=== Comparación (test) ===")
    print(f"{'modelo':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>8}{'thr':>7}")
    for r in resumen:
        print(f"{r['modelo']:<26}{r['roc_auc']:>9}{r['pr_auc']:>9}"
              f"{r['f1_desiste']:>8}{r['threshold']:>7}")

    print("\n=== Reporte final (LightGBM calibrado, umbral "
          f"{best_threshold:.3f}) ===")
    print(classification_report(y_test, pred_test, digits=3))
    print("Matriz de confusión [ [VN FP] [FN VP] ]:")
    print(confusion_matrix(y_test, pred_test))

    # ---- Guardar artefacto (compatible con app.py) ----
    artifact = {
        "modelo": calibrado,
        "best_threshold": best_threshold,
        "feature_cols": X.columns.tolist(),
    }
    joblib.dump(artifact, "modelo_desistimiento_lgbm.joblib")

    final = next(r for r in resumen if r["modelo"] == "LightGBM calibrado")
    rep = classification_report(y_test, pred_test, output_dict=True)
    metrics = {
        "comparacion_test": resumen,
        "final": {
            "roc_auc": final["roc_auc"],
            "pr_auc": final["pr_auc"],
            "threshold": round(best_threshold, 4),
            "recall_desiste": round(rep["1"]["recall"], 4),
            "precision_desiste": round(rep["1"]["precision"], 4),
            "f1_desiste": round(rep["1"]["f1-score"], 4),
            "accuracy": round(rep["accuracy"], 4),
        },
    }
    with open("metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print("\nArtefacto guardado: modelo_desistimiento_lgbm.joblib")
    print("Métricas guardadas: metrics.json")


if __name__ == "__main__":
    main()
