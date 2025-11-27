# streamlit_app.py (VERSIÓN CON HORA ACTUAL)
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# ==========================
# Configuración de la página
# ==========================
st.set_page_config(
    page_title="Predicción de Entregas a Tiempo",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Predicción de Entregas a Tiempo")
st.write("Predice si una entrega llegará a tiempo basándose en condiciones previas al envío.")

# ==========================
# 1. Cargar el modelo
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
    st.markdown("#### 🌤️ Condiciones de la Ruta")
    clima = st.selectbox(
        "Clima",
        options=["Bueno", "Lluvia", "Tormenta"],
        help="Condiciones meteorológicas esperadas"
    )

    trafico = st.selectbox(
        "Nivel de Tráfico",
        options=["Bajo", "Medio", "Alto"],
        help="Congestión vehicular esperada"
    )

    riesgo_ruta = st.selectbox(
        "Riesgo de Ruta",
        options=["Bajo", "Medio", "Alto"],
        help="Nivel de riesgo de la ruta (seguridad, condiciones del camino)"
    )

    distancia_km = st.number_input(
        "Distancia Total (km)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
        format="%.2f",
        help="Distancia total de la ruta"
    )

with col2:
    st.markdown("#### 📦 Información de la Carga")
    tipo_carga = st.selectbox(
        "Tipo de Carga",
        options=["Normal", "Fragil", "Peligrosa"]
    )

    peso_kg = st.number_input(
        "Peso de Carga (kg)",
        min_value=0,
        max_value=20000,
        value=1000,
        step=100,
        help="Peso total de la carga"
    )

    # VENTANA DE TIEMPO
    st.markdown("##### ⏰ Ventana de Entrega")
    hora_inicio_entrega = st.time_input(
        "Hora inicio de ventana de entrega",
        value=pd.Timestamp("09:00").time(),
        help="Primera hora en que se puede entregar"
    )
    
    hora_fin_entrega = st.time_input(
        "Hora fin de ventana de entrega",
        value=pd.Timestamp("11:00").time(),
        help="Última hora en que se puede entregar"
    )

with col3:
    st.markdown("#### 🚛 Conductor y Vehículo")
    experiencia = st.number_input(
        "Experiencia del Conductor (años)",
        min_value=0,
        max_value=40,
        value=5,
        help="Años de experiencia en entregas"
    )

    antiguedad_camion = st.number_input(
        "Antigüedad del Vehículo (años)",
        min_value=0,
        max_value=30,
        value=3,
        help="Edad del vehículo"
    )

    fallas_mecanicas = st.selectbox(
        "Historial de Fallas Mecánicas",
        options=["No", "Si"],
        help="¿El vehículo ha tenido fallas recientes?"
    )

    nivel_combustible = st.number_input(
        "Nivel de Combustible Inicial (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=5.0,
        format="%.1f"
    )

st.markdown("---")

# ==========================
# 3. CALCULAR TIEMPO ESTIMADO
# ==========================
VELOCIDAD_BASE = 40  # km/h

factores_clima = {"Bueno": 1.0, "Lluvia": 0.8, "Tormenta": 0.5}
factores_trafico = {"Bajo": 1.0, "Medio": 0.75, "Alto": 0.5}

def calcular_tiempo_estimado(distancia, clima, trafico, experiencia, antiguedad):
    """Calcula tiempo estimado similar al código TypeScript"""
    velocidad_efectiva = VELOCIDAD_BASE * factores_clima[clima] * factores_trafico[trafico]
    
    # Ajustar por experiencia
    if experiencia < 2:
        velocidad_efectiva *= 0.7  # Junior
    elif experiencia < 5:
        velocidad_efectiva *= 0.85  # Intermedio
    else:
        velocidad_efectiva *= 1.0  # Senior
    
    # Ajustar por antigüedad del vehículo
    if antiguedad > 10:
        velocidad_efectiva *= 0.85
    elif antiguedad > 5:
        velocidad_efectiva *= 0.9
    
    tiempo_viaje_horas = distancia / velocidad_efectiva
    tiempo_minutos = tiempo_viaje_horas * 60
    tiempo_parada = 15  # minutos por entrega
    
    return tiempo_minutos + tiempo_parada

def determinar_horario_salida(hora_actual):
    """Determina el horario de salida basado en la hora actual"""
    hora = hora_actual.hour
    
    if 6 <= hora < 12:
        return "Manana"
    elif 12 <= hora < 18:
        return "Tarde"
    else:
        return "Noche"

# ==========================
# 4. Botón de predicción
# ==========================
if st.button("🔮 Predecir Entrega", type="primary", use_container_width=True):
    
    # OBTENER HORA ACTUAL
    hora_actual = datetime.now()
    
    # DETERMINAR HORARIO DE SALIDA
    horario = determinar_horario_salida(hora_actual)
    
    # CALCULAR TIEMPO ESTIMADO
    tiempo_estimado = calcular_tiempo_estimado(
        distancia_km, clima, trafico, experiencia, antiguedad_camion
    )
    
    # CALCULAR HORA ESTIMADA DE LLEGADA
    hora_llegada_estimada = hora_actual + timedelta(minutes=tiempo_estimado)
    
    # CREAR DATETIME PARA LA HORA INICIO DE VENTANA
    hora_inicio_ventana = datetime.combine(hora_actual.date(), hora_inicio_entrega)
    hora_fin_ventana = datetime.combine(hora_actual.date(), hora_fin_entrega)
    
    # Si la ventana es para el día siguiente (ej: es de noche y la ventana es mañana)
    if hora_inicio_ventana < hora_actual:
        hora_inicio_ventana += timedelta(days=1)
        hora_fin_ventana += timedelta(days=1)
    
    # CALCULAR DEMORA: (Hora Actual + Tiempo Estimado) - Hora Inicio Ventana
    # Demora positiva = llega tarde, negativa = llega antes
    demora_minutos = (hora_llegada_estimada - hora_inicio_ventana).total_seconds() / 60
    
    # CALCULAR TIEMPO REAL (simulado como tiempo estimado + algo de variación)
    tiempo_real_simulado = tiempo_estimado + (demora_minutos if demora_minutos > 0 else 0)
    
    # Construir DataFrame
    nueva_entrada = pd.DataFrame([{
        "Clima": clima,
        "TraficoPico": trafico,
        "RiesgoRuta": riesgo_ruta,
        "Distancia_km": distancia_km,
        "TiempoEstimado_min": tiempo_estimado,
        "TiempoReal_min": tiempo_real_simulado,
        "Demora_min": demora_minutos,  # Solo valores positivos para la demora
        "TipoCarga": tipo_carga,
        "Peso_kg": peso_kg,
        "ExperienciaConductor_anios": experiencia,
        "AntiguedadCamion_anios": antiguedad_camion,
        "FallasMecanicas": fallas_mecanicas,
        "NivelCombustible_pct": nivel_combustible,
        "HorarioSalida": horario,
    }])

    # Mostrar información de tiempos
    st.subheader("📊 Análisis de Tiempos")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("🕐 Hora Actual", hora_actual.strftime("%H:%M"))
        st.caption(f"Horario: {horario}")
    
    with col_b:
        st.metric("⏱️ Tiempo Estimado", f"{tiempo_estimado:.1f} min")
    
    with col_c:
        st.metric("🎯 Llegada Estimada", hora_llegada_estimada.strftime("%H:%M"))
    
    with col_d:
        demora_display = f"{demora_minutos:.1f} min"
        delta_color = "off" if demora_minutos <= 0 else "inverse"
        st.metric("📊 Demora", demora_display,
                 delta="A tiempo" if demora_minutos <= 0 else "Retrasado",
                 delta_color=delta_color)
    
    # Info de ventana de entrega
    st.info(f"🎯 **Ventana de entrega:** {hora_inicio_entrega.strftime('%H:%M')} - {hora_fin_entrega.strftime('%H:%M')}")
    
    # Análisis de margen
    if demora_minutos <= 0:
        st.success("✅ Llegará a tiempo")
    else:
        st.error("❌ Llegará tarde")

st.markdown("---")
st.caption("🔧 Sistema de Predicción de Entregas v2.1 | Hora actual: " + datetime.now().strftime("%H:%M:%S"))