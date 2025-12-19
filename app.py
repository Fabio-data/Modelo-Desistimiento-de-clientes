import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Config
# =========================
st.set_page_config(
    page_title="Desistimiento de Clientes",
    page_icon="🔍",
    layout="wide"
)

def fmt_cop(x: float) -> str:
    # 1.234.567 COP
    return f"$ {x:,.0f}".replace(",", ".") + " COP"

@st.cache_resource
def cargar_artifact():
    artifact = joblib.load("modelo_desistimiento_lgbm.joblib")
    return artifact["modelo"], float(artifact["best_threshold"]), artifact["feature_cols"]

modelo, best_threshold, feature_cols = cargar_artifact()

# =========================
# UI - Header
# =========================
st.title("🔍 Probabilidad de Desistimiento de Clientes")
st.write(
    "Ingresa datos básicos del cliente y el modelo estimará la **probabilidad de desistir** "
    "durante el proceso de solicitud de crédito.\n\n"
    f"**Umbral operativo del modelo:** `{best_threshold:.2f}` (si la probabilidad ≥ umbral, se marca como riesgo)."
)

# =========================
# Sidebar Inputs (COP)
# =========================
st.sidebar.header("Parámetros del cliente (COP)")

ingresos = st.sidebar.number_input(
    "Ingresos mensuales",
    min_value=0.0,
    value=2_000_000.0,
    step=200_000.0,
    format="%.0f"
)
st.sidebar.caption(f"Ingresos: {fmt_cop(ingresos)}")

egresos = st.sidebar.number_input(
    "Egresos mensuales",
    min_value=0.0,
    value=1_000_000.0,
    step=200_000.0,
    format="%.0f"
)
st.sidebar.caption(f"Egresos: {fmt_cop(egresos)}")

valor_solicitado = st.sidebar.number_input(
    "Valor del crédito solicitado",
    min_value=0.0,
    value=3_000_000.0,
    step=500_000.0,
    format="%.0f"
)
st.sidebar.caption(f"Crédito solicitado: {fmt_cop(valor_solicitado)}")

personas_cargo = st.sidebar.number_input(
    "Personas a cargo",
    min_value=0,
    value=1,
    step=1
)

tipo_contrato = st.sidebar.selectbox(
    "Tipo de contrato",
    options=["Indefinido", "Término fijo", "Temporal", "Otra"],
    index=3
)

# =========================
# Validaciones / Alertas
# =========================
if ingresos <= 0:
    st.warning("⚠️ Ingresos en 0 distorsiona ratios (endeudamiento/solicitud). Ajusta para una predicción más confiable.")
if egresos > ingresos and ingresos > 0:
    st.info("ℹ️ Egresos mayores que ingresos: esto suele incrementar el riesgo. Verifica si el dato es correcto.")

st.write("Completa los datos en la izquierda y presiona **Calcular**.")

# =========================
# Predict
# =========================
if st.button("Calcular probabilidad de desistimiento", type="primary"):
    # Construimos solo lo necesario y respetamos feature_cols
    row = {}

    # Base
    if "INGRESOS" in feature_cols: row["INGRESOS"] = ingresos
    if "EGRESOS" in feature_cols: row["EGRESOS"] = egresos
    if "VALOR_SOLICITADO" in feature_cols: row["VALOR_SOLICITADO"] = valor_solicitado
    if "PERSONAS_CARGO" in feature_cols: row["PERSONAS_CARGO"] = personas_cargo
    if "TIPO_CONTRATO" in feature_cols: row["TIPO_CONTRATO"] = tipo_contrato

    # Features derivadas (según tu entrenamiento)
    cap_pago = ingresos - egresos
    ratio_endeud = egresos / (ingresos + 1e-6)
    ratio_sol_ing = valor_solicitado / (ingresos + 1e-6)
    estres = ratio_endeud + ratio_sol_ing

    if "CAPACIDAD_PAGO" in feature_cols: row["CAPACIDAD_PAGO"] = cap_pago
    if "RATIO_ENDEUDAMIENTO" in feature_cols: row["RATIO_ENDEUDAMIENTO"] = ratio_endeud
    if "RATIO_SOLICITUD_INGRESO" in feature_cols: row["RATIO_SOLICITUD_INGRESO"] = ratio_sol_ing
    # Nota: en tu entrenamiento la columna se llama ESTRES_FINANCIEROO (doble O)
    if "ESTRES_FINANCIEROO" in feature_cols: row["ESTRES_FINANCIEROO"] = estres

    # DataFrame con columnas esperadas (en el orden correcto)
    fila = pd.DataFrame([{col: row.get(col, np.nan) for col in feature_cols}])

    # Predicción
    proba = float(modelo.predict_proba(fila)[0, 1])
    pred = int(proba >= best_threshold)

    # Riesgo alineado al umbral (no cortes arbitrarios)
    low_cut = best_threshold * 0.75
    high_cut = best_threshold * 1.25
    if proba < low_cut:
        nivel = "Bajo"
        accion = "Flujo normal (sin intervención inmediata)."
    elif proba < high_cut:
        nivel = "Medio"
        accion = "Seguimiento recomendado (contacto en 24h / revisar oferta)."
    else:
        nivel = "Alto"
        accion = "Intervención prioritaria (contacto rápido / oferta asistida)."

    # =========================
    # UI Results
    # =========================
    st.subheader("Resultado del modelo")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Probabilidad de DESISTIR", f"{proba*100:.1f}%")
    with col2:
        st.metric("Nivel de riesgo", nivel)
    with col3:
        st.write("**Acción sugerida:**")
        st.write(accion)

    if pred == 1:
        st.error("⚠️ El modelo estima **ALTA PROBABILIDAD DE DESISTIR** (por encima del umbral).")
    else:
        st.success("✅ El modelo estima que el cliente **PROBABLEMENTE NO DESISTE** (por debajo del umbral).")

    st.caption(f"Umbral usado por el modelo: {best_threshold:.2f}")

    # Explicación rápida (señales principales)
    st.write("### Señales principales (para interpretación rápida)")
    st.write(f"- **Capacidad de pago:** {fmt_cop(cap_pago)}")
    st.write(f"- **Ratio endeudamiento (egresos/ingresos):** {ratio_endeud*100:.1f}%")
    st.write(f"- **Ratio solicitud/ingreso:** {ratio_sol_ing*100:.1f}%")
    st.write(f"- **Estrés financiero (suma de ratios):** {estres*100:.1f}%")

    with st.expander("Ver datos enviados al modelo (debug)"):
        st.dataframe(fila)
