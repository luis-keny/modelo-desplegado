# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model

# ============================================================
# 1. Cargar modelo y scaler
# ============================================================
ARTEFACTOS_DIR = Path("artefactos")
MODEL_PATH = ARTEFACTOS_DIR / "mlp_model.keras"
SCALER_PATH = ARTEFACTOS_DIR / "scaler.pkl"

# Modelo Keras
model = load_model(MODEL_PATH)

# Scaler (StandardScaler)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Umbral óptimo
BEST_THRESHOLD = 0.74

# ============================================================
# 2. Configuración de la app
# ============================================================
st.set_page_config(
    page_title="Predicción de Entrega a Tiempo",
    layout="wide"
)

st.title("🚚 Predicción de Entrega a Tiempo")
st.write("Modelo MLP para predecir si una entrega llegará a tiempo o con demora.")

tab1, tab2, tab3 = st.tabs(
    ["🔮 Predicción", "📊 Información del modelo", "📈 Gráficos del entrenamiento"]
)

# ============================================================
# 3. Mapeos de variables categóricas
#    (ajusta estos códigos si en tu entrenamiento usaste otros)
# ============================================================
MAP_CLIMA = {
    "Bueno": 0,      # equivalente a "Despejado"
    "Lluvia": 1,
    "Tormenta": 2,
}

MAP_TRAFICO = {
    "Bajo": 0,       # sin tráfico / bajo
    "Medio": 1,      # moderado
    "Alto": 2,       # pesado
}

MAP_RIESGO = {
    "Bajo": 0,
    "Medio": 1,
    "Alto": 2,
}

# Ajusta si en tu dataset final usaste otra codificación
MAP_TIPO_CARGA = {
    "Normal": 0,
    "Frágil": 1,
    "Peligrosa": 2,
}

MAP_FALLAS = {
    "No": 0,
    "Sí": 1,
}

MAP_HORARIO = {
    "Mañana": 0,
    "Tarde": 1,
    "Noche": 2,
}

# ============================================================
# TAB 1: Formulario de predicción
# ============================================================
with tab1:
    st.subheader("Ingresar datos del viaje")

    with st.form("form_prediccion"):
        col1, col2, col3 = st.columns(3)

        with col1:
            clima = st.selectbox("Clima", ["Bueno", "Lluvia", "Tormenta"])
            trafico = st.selectbox("Tráfico en hora pico", ["Bajo", "Medio", "Alto"])
            riesgo = st.selectbox("Riesgo de la ruta", ["Bajo", "Medio", "Alto"])
            horario = st.selectbox("Horario de salida", ["Mañana", "Tarde", "Noche"])

        with col2:
            distancia = st.number_input(
                "Distancia (km)",
                min_value=0.0,
                max_value=2000.0,
                value=200.0,
                step=1.0
            )
            tiempo_estimado = st.number_input(
                "Tiempo estimado (min)",
                min_value=0.0,
                max_value=2000.0,
                value=300.0,
                step=1.0
            )
            tiempo_real = st.number_input(
                "Tiempo real esperado (min)",
                min_value=0.0,
                max_value=2000.0,
                value=320.0,
                step=1.0
            )
            # Se puede recalcular Demora automáticamente
            calcula_demora = st.checkbox("Calcular demora automáticamente", value=True)
            if calcula_demora:
                demora = max(tiempo_real - tiempo_estimado, 0.0)
            else:
                demora = st.number_input(
                    "Demora (min)",
                    min_value=-500.0,
                    max_value=500.0,
                    value=20.0,
                    step=1.0
                )

        with col3:
            tipo_carga = st.selectbox("Tipo de carga", ["Normal", "Frágil", "Peligrosa"])
            peso = st.number_input(
                "Peso de la carga (kg)",
                min_value=0.0,
                max_value=50000.0,
                value=8000.0,
                step=100.0
            )
            experiencia = st.number_input(
                "Experiencia del conductor (años)",
                min_value=0,
                max_value=50,
                value=5,
                step=1
            )
            antig_camion = st.number_input(
                "Antigüedad del camión (años)",
                min_value=0,
                max_value=40,
                value=5,
                step=1
            )
            fallas = st.selectbox("¿Hubo fallas mecánicas?", ["No", "Sí"])
            nivel_comb = st.slider(
                "Nivel de combustible al inicio (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0
            )

        st.markdown(f"**Demora calculada:** {demora:.1f} min" if calcula_demora else "")

        submitted = st.form_submit_button("Predecir entrega")

    if submitted:
        # --------------------------------------------------------
        # 1) Codificar variables categóricas
        # --------------------------------------------------------
        features = {
            "Clima": MAP_CLIMA[clima],
            "TraficoPico": MAP_TRAFICO[trafico],
            "RiesgoRuta": MAP_RIESGO[riesgo],
            "Distancia_km": distancia,
            "TiempoEstimado_min": tiempo_estimado,
            "TiempoReal_min": tiempo_real,
            "Demora_min": demora,
            "TipoCarga": MAP_TIPO_CARGA[tipo_carga],
            "Peso_kg": peso,
            "ExperienciaConductor_anios": experiencia,
            "AntiguedadCamion_anios": antig_camion,
            "FallasMecanicas": MAP_FALLAS[fallas],
            "NivelCombustible_pct": nivel_comb,
            "HorarioSalida": MAP_HORARIO[horario],
            # Si tu scaler/modelo tiene más columnas derivadas,
            # deberás agregarlas aquí con los mismos nombres.
        }

        X_input = pd.DataFrame([features])

        # Alinear columnas con las que espera el scaler (si tiene feature_names_in_)
        try:
            X_input = X_input[scaler.feature_names_in_]
        except AttributeError:
            pass

        # --------------------------------------------------------
        # 2) Escalar e inferir
        # --------------------------------------------------------
        X_scaled = scaler.transform(X_input)
        prob = float(model.predict(X_scaled)[0][0])
        pred_bin = int(prob >= BEST_THRESHOLD)

        # --------------------------------------------------------
        # 3) Mostrar resultados
        # --------------------------------------------------------
        st.subheader("Resultado de la predicción")

        if pred_bin == 1:
            st.success("✅ Predicción: **ENTREGA A TIEMPO**")
        else:
            st.error("⚠️ Predicción: **ENTREGA CON DEMORA**")

        st.write(f"Probabilidad de **llegar a tiempo** (salida sigmoide): **{prob:.3f}**")
        st.write(f"Umbral usado para clasificar: **{BEST_THRESHOLD:.2f}**")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.write("**Variables ingresadas (ya codificadas):**")
            st.dataframe(X_input)

        with col_r2:
            st.write("**Vector de entrada escalado (primeros valores):**")
            st.write(pd.DataFrame(X_scaled, columns=X_input.columns).head())


# ============================================================
# TAB 3: Gráficos del entrenamiento
# ============================================================
with tab3:
    st.subheader("Gráficos del entrenamiento")

    hist_path = ARTEFACTOS_DIR / "mlp_training_history.png"
    weights_path = ARTEFACTOS_DIR / "mlp_weights_distribution.png"
    arch_path = ARTEFACTOS_DIR / "mlp_architecture.png"

    colg1, colg2 = st.columns(2)

    if hist_path.exists():
        with colg1:
            st.image(str(hist_path), caption="Histórico de entrenamiento (loss, AUC, etc.)", use_column_width=True)
    else:
        st.write("No se encontró `mlp_training_history.png`.")

    if weights_path.exists():
        with colg2:
            st.image(str(weights_path), caption="Distribución de pesos del modelo", use_column_width=True)
    else:
        st.write("No se encontró `mlp_weights_distribution.png`.")

    if arch_path.exists():
        st.image(str(arch_path), caption="Arquitectura del modelo MLP", use_column_width=True)
    else:
        st.write("No se encontró `mlp_architecture.png`.")
