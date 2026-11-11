"""Preparación de datos compartida por train.py y app.py.

Una sola definición de las variables derivadas evita que el entrenamiento y la
app calculen cosas distintas.
"""
import numpy as np
import pandas as pd

TARGET = "DESISTE"
ESTADOS_FINALES = ["Desistida", "Negada", "Aprobada", "Anulada"]
# Identificadores, fecha y marca no aportan señal; GENERO se excluye para no
# decidir con un atributo protegido.
COLUMNAS_DESCARTADAS = ["SOLICITUD", "FECHA_INICIO", "GENERO", "Marca producto"]
DERIVADAS = [
    "CAPACIDAD_PAGO",
    "RATIO_ENDEUDAMIENTO",
    "RATIO_SOLICITUD_INGRESO",
    "ESTRES_FINANCIERO",
]
_EPS = 1e-6  # evita dividir por cero cuando INGRESOS = 0


def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de `df` con las variables de negocio derivadas."""
    df = df.copy()
    df["CAPACIDAD_PAGO"] = df["INGRESOS"] - df["EGRESOS"]
    df["RATIO_ENDEUDAMIENTO"] = df["EGRESOS"] / (df["INGRESOS"] + _EPS)
    df["RATIO_SOLICITUD_INGRESO"] = df["VALOR_SOLICITADO"] / (df["INGRESOS"] + _EPS)
    df["ESTRES_FINANCIERO"] = df["RATIO_ENDEUDAMIENTO"] + df["RATIO_SOLICITUD_INGRESO"]
    return df


def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Limpia el dataset crudo y devuelve (X, y) listos para el pipeline."""
    df = df[df["Estado"].isin(ESTADOS_FINALES)].copy()
    y = (df["Estado"] == "Desistida").astype(int).rename(TARGET)

    df = df.drop(columns=[*COLUMNAS_DESCARTADAS, "Estado"])
    df["TIPO_CONTRATO"] = df["TIPO_CONTRATO"].fillna("Otra")
    # Edades imposibles (-1, 2000...) pasan a nulo; el pipeline las imputa.
    df["EDAD"] = df["EDAD"].where(df["EDAD"].between(18, 100), np.nan)

    return agregar_variables(df), y
