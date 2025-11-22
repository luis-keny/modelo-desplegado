# streamlit_app.py
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

# ==========================
# Configuración de la página
# ==========================
st.set_page_config(
    page_title="Predicción de Entregas a Tiempo",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Predicción de Entregas a Tiempo")
st.write("Aplicación para predecir si una entrega llegará a tiempo usando un modelo MLP entrenado.")

# ==========================
# 1. Cargar el modelo (pipeline completo)
# ==========================
MODEL_PATH = Path("artefactos/modelo_entregas_mlp.pkl")

try:
    pipe = joblib.load(MODEL_PATH)
    st.success("✅ Modelo cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# ==========================
# 2. Formulario de entrada
# ==========================
st.markdown("### 📝 Ingrese los datos del envío")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🌤️ Condiciones Ambientales")
    clima = st.selectbox(
        "Clima",
        options=["Bueno", "Lluvia", "Tormenta"],
        help="Condiciones meteorológicas durante la entrega"
    )

    trafico = st.selectbox(
        "Nivel de Tráfico (TraficoPico)",
        options=["Bajo", "Medio", "Alto"],
        help="Congestión vehicular en la ruta"
    )

    riesgo_ruta = st.selectbox(
        "Riesgo de Ruta",
        options=["Bajo", "Medio", "Alto"],
        help="Riesgo asociado a la ruta de entrega"
    )

    horario = st.selectbox(
        "Horario de Salida",
        options=["Manana", "Tarde", "Noche"]
    )

with col2:
    st.markdown("#### 📦 Información de la Carga y Tiempos")
    tipo_carga = st.selectbox(
        "Tipo de Carga",
        options=["Normal", "Fragil", "Peligrosa"]
    )

    peso_kg = st.number_input(
        "Peso de Carga (kg)",
        min_value=0,
        max_value=20000,
        value=4013,
        step=100
    )

    distancia_km = st.number_input(
        "Distancia (km)",
        min_value=0.0,
        max_value=1000.0,
        value=291.94,
        step=1.0,
        format="%.2f"
    )

    tiempo_estimado = st.number_input(
        "Tiempo Estimado (min)",
        min_value=0.0,
        max_value=1000.0,
        value=291.9,
        step=1.0,
        format="%.1f"
    )

    tiempo_real = st.number_input(
        "Tiempo Real (min)",
        min_value=0.0,
        max_value=1000.0,
        value=327.5,
        step=1.0,
        format="%.1f"
    )

    demora = st.number_input(
        "Demora (min)",
        min_value=-100.0,
        max_value=300.0,
        value=35.6,
        step=1.0,
        format="%.1f"
    )

with col3:
    st.markdown("#### 🚛 Información del Conductor y Vehículo")
    experiencia = st.number_input(
        "Experiencia del Conductor (años)",
        min_value=0,
        max_value=40,
        value=5
    )

    antiguedad_camion = st.number_input(
        "Antigüedad del Camión (años)",
        min_value=0,
        max_value=30,
        value=6
    )

    fallas_mecanicas = st.selectbox(
        "Fallas Mecánicas",
        options=["No", "Si"]
    )

    nivel_combustible = st.number_input(
        "Nivel de Combustible (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.6,
        step=0.1,
        format="%.1f"
    )

st.markdown("---")

# ==========================
# 3. Botón de predicción
# ==========================
if st.button("🔮 Predecir Entrega", type="primary", use_container_width=True):
    # Construir DataFrame exactamente como el código de prueba
    nueva_entrada = pd.DataFrame([{
        "Clima": clima,
        "TraficoPico": trafico,
        "RiesgoRuta": riesgo_ruta,
        "Distancia_km": distancia_km,
        "TiempoEstimado_min": tiempo_estimado,
        "TiempoReal_min": tiempo_real,
        "Demora_min": demora,
        "TipoCarga": tipo_carga,
        "Peso_kg": peso_kg,
        "ExperienciaConductor_anios": experiencia,
        "AntiguedadCamion_anios": antiguedad_camion,
        "FallasMecanicas": fallas_mecanicas,
        "NivelCombustible_pct": nivel_combustible,
        "HorarioSalida": horario,
    }])

    st.subheader("📊 Datos de entrada")
    st.dataframe(nueva_entrada, use_container_width=True)

    try:
        pred = pipe.predict(nueva_entrada)
        prob = pipe.predict_proba(nueva_entrada)

        # Asumimos que la clase 1 es "Entrega a tiempo"
        pred_clase = int(pred[0])
        prob_clase_1 = float(prob[0][1])

        st.markdown("---")
        st.subheader("🧠 Resultado del modelo")

        if pred_clase == 1:
            st.success("✅ Predicción: **Entrega a tiempo (clase 1)**")
        else:
            st.error("⚠️ Predicción: **Entrega con retraso (clase 0)**")

        st.metric(
            "Probabilidad clase 1 (Entrega a tiempo)",
            f"{prob_clase_1 * 100:.2f} %"
        )

        st.progress(prob_clase_1)

        # Mostrar también ambas probabilidades
        st.write("Distribución de probabilidades por clase:")
        st.write(
            pd.DataFrame(
                {
                    "Clase": [0, 1],
                    "Probabilidad": prob[0]
                }
            )
        )

    except Exception as e:
        st.error(f"❌ Error al realizar la predicción: {e}")
        st.info("Verifica que las columnas del DataFrame coincidan con las que el pipeline fue entrenado.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p>🔧 Predicción de Entregas v1.0 | MLP + Pipeline de Preprocesamiento</p>
</div>
""",
    unsafe_allow_html=True
)
