"""App de Streamlit: estima la probabilidad de que un cliente desista."""
import joblib
import pandas as pd
import streamlit as st

from features import agregar_variables

ARTIFACT_PATH = "modelo_desistimiento_lgbm.joblib"

# Variables que siempre se piden: etiqueta y paso del control numérico.
PRINCIPALES = {
    "INGRESOS": ("Ingresos mensuales (COP)", 200_000.0),
    "EGRESOS": ("Egresos mensuales (COP)", 200_000.0),
    "VALOR_SOLICITADO": ("Valor del crédito solicitado (COP)", 500_000.0),
    "PERSONAS_CARGO": ("Personas a cargo", 1.0),
    "TIPO_CONTRATO": ("Tipo de contrato", None),
}

st.set_page_config(page_title="Desistimiento de Clientes", page_icon="🔍", layout="wide")


@st.cache_resource
def cargar_artifact() -> dict:
    return joblib.load(ARTIFACT_PATH)


def fmt_cop(x: float) -> str:
    return f"$ {x:,.0f}".replace(",", ".") + " COP"


artifact = cargar_artifact()
modelo = artifact["modelo"]
umbral = float(artifact["best_threshold"])
tipicos = artifact["valores_tipicos"]
categorias = artifact["categorias"]


def pedir(col: str, contenedor, etiqueta: str | None = None, paso: float = 1.0):
    """Control de entrada para `col`, prellenado con su valor típico."""
    etiqueta = etiqueta or col.replace("_", " ").capitalize()
    if col in categorias:
        opciones = categorias[col]
        return contenedor.selectbox(etiqueta, opciones, index=opciones.index(tipicos[col]))
    return contenedor.number_input(
        etiqueta, value=float(tipicos[col]), step=paso, format="%.0f")


st.title("🔍 Probabilidad de Desistimiento de Clientes")
st.write(
    "Ingresa los datos del cliente y el modelo estimará la **probabilidad de desistir** "
    "durante la solicitud de crédito. "
    f"Si la probabilidad supera el umbral operativo (`{umbral:.2f}`), se marca como riesgo."
)
st.caption(
    "Demo educativa con **datos ficticios**: no debe usarse para decisiones reales de "
    "crédito. Las variables que no cambies usan el valor típico del conjunto de entrenamiento."
)

# ---- Entradas ----
st.sidebar.header("Datos del cliente")
valores = {
    col: pedir(col, st.sidebar, etiqueta, paso or 1.0)
    for col, (etiqueta, paso) in PRINCIPALES.items()
}
with st.expander("Más variables (opcional)"):
    columnas = st.columns(3)
    resto = [c for c in tipicos if c not in PRINCIPALES]
    for i, col in enumerate(resto):
        valores[col] = pedir(col, columnas[i % 3])

ingresos, egresos = valores["INGRESOS"], valores["EGRESOS"]
if ingresos <= 0:
    st.warning("Con ingresos en 0 los ratios se disparan y la predicción no es confiable.")
elif egresos > ingresos:
    st.info("Los egresos superan a los ingresos: verifica que el dato sea correcto.")

# ---- Predicción ----
if st.button("Calcular probabilidad de desistimiento", type="primary"):
    fila = agregar_variables(pd.DataFrame([valores]))[artifact["feature_cols"]]
    proba = float(modelo.predict_proba(fila)[0, 1])

    # Niveles de riesgo relativos al umbral del modelo, no cortes arbitrarios.
    if proba < umbral * 0.75:
        nivel, accion = "Bajo", "Flujo normal (sin intervención inmediata)."
    elif proba < umbral * 1.25:
        nivel, accion = "Medio", "Seguimiento recomendado (contacto en 24 h / revisar oferta)."
    else:
        nivel, accion = "Alto", "Intervención prioritaria (contacto rápido / oferta asistida)."

    st.subheader("Resultado del modelo")
    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("Probabilidad de desistir", f"{proba:.1%}")
    col2.metric("Nivel de riesgo", nivel)
    col3.write(f"**Acción sugerida:** {accion}")

    if proba >= umbral:
        st.error("⚠️ Por encima del umbral: **alta probabilidad de desistir**.")
    else:
        st.success("✅ Por debajo del umbral: el cliente **probablemente no desiste**.")

    datos = fila.iloc[0]
    st.write("### Señales principales")
    st.write(f"- **Capacidad de pago:** {fmt_cop(datos['CAPACIDAD_PAGO'])}")
    st.write(f"- **Egresos / ingresos:** {datos['RATIO_ENDEUDAMIENTO']:.1%}")
    st.write(f"- **Crédito solicitado / ingresos:** {datos['RATIO_SOLICITUD_INGRESO']:.1%}")

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(fila)
