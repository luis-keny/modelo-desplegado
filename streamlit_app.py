# streamlit_app.py (VERSIÓN COMPLETA CON PREDICCIÓN Y CLUSTERING)
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# Configuración de la página
# ==========================
st.set_page_config(
    page_title="Sistema de Análisis de Entregas",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Sistema de Análisis de Entregas")

# ==========================
# Selector de Módulo
# ==========================
st.markdown("### 📊 Seleccione el módulo de análisis")
modulo = st.radio(
    "",
    options=["🔮 Predicción de Entregas", "📈 Clustering + PCA de Conductores"],
    horizontal=True
)

st.markdown("---")

# ==========================
# MÓDULO 1: PREDICCIÓN
# ==========================
if modulo == "🔮 Predicción de Entregas":
    st.write("Predice si una entrega llegará a tiempo basándose en condiciones previas al envío.")
    
    # Cargar el modelo
    MODEL_PATH = Path("artefactos/modelo_entregas_mlp.pkl")
    
    try:
        pipe = joblib.load(MODEL_PATH)
        st.success("✅ Modelo cargado exitosamente")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()
    
    # ==========================
    # CASOS PREDEFINIDOS
    # ==========================
    casos_predefinidos = {
        "Caso 1: Ruta Corta - Condiciones Óptimas": {
            "descripcion": "Entrega de 50 km en condiciones ideales con amplia ventana de entrega",
            "datos": {
                "clima": "Bueno",
                "trafico": "Bajo",
                "riesgo_ruta": "Bajo",
                "distancia_km": 50.0,
                "tipo_carga": "Normal",
                "peso_kg": 700,
                "hora_inicio_entrega": pd.Timestamp("16:30").time(),
                "hora_fin_entrega": pd.Timestamp("21:45").time(),
                "experiencia": 5,
                "antiguedad_camion": 3,
                "fallas_mecanicas": "No",
                "nivel_combustible": 90.0
            },
            "contexto": {
                "conductor": "C045",
                "frenadas_duras": 18,
                "excesos_velocidad": 5,
                "infracciones": 2,
                "incidentes_carga": 1,
                "accidentes_leves": 1,
                "reclamos": 3,
                "horas_mes": 190,
                "km_mes": 8500,
                "entregas_mes": 95,
                "asistencia_capacitacion": "75%",
                "indice_fatiga": 0.68
            },
            "recomendaciones": [
                "**Conducción**: Mantener ruta y horario actuales con conducción suave; evitar acelerar innecesariamente",
                "**Ventana de tiempo**: Aprovechar la holgura de la ventana (12:30–21:45); no hay presión de reloj",
                "**Planificación**: Mantener programación actual, sin cambios de ruta ni horarios",
                "**Gestión del conductor**: Monitorear frenadas duras, excesos y reclamos; usar esta ruta simple para corregir hábitos con baja presión",
                "**Capacitación**: Programar capacitación en conducción defensiva, gestión de fatiga y cuidado de carga"
            ],
            "riesgos": {
                "servicio": "Bajo",
                "seguridad": "Medio (por estilo y fatiga)",
                "mecanico": "Bajo",
                "global": "Bajo"
            }
        },
        "Caso 2: Ruta Larga - Clima Adverso": {
            "descripcion": "Entrega de 200 km con lluvia, tráfico alto y ventana de entrega ajustada",
            "datos": {
                "clima": "Lluvia",
                "trafico": "Alto",
                "riesgo_ruta": "Bajo",
                "distancia_km": 200.0,
                "tipo_carga": "Normal",
                "peso_kg": 1000,
                "hora_inicio_entrega": pd.Timestamp("16:30").time(),
                "hora_fin_entrega": pd.Timestamp("19:00").time(),
                "experiencia": 5,
                "antiguedad_camion": 3,
                "fallas_mecanicas": "No",
                "nivel_combustible": 80.0
            },
            "contexto": {
                "conductor": "C045",
                "frenadas_duras": 18,
                "excesos_velocidad": 5,
                "infracciones": 2,
                "incidentes_carga": 1,
                "accidentes_leves": 1,
                "reclamos": 3,
                "horas_mes": 190,
                "km_mes": 8500,
                "entregas_mes": 95,
                "asistencia_capacitacion": "75%",
                "indice_fatiga": 0.68
            },
            "recomendaciones": [
                "**Conducción**: Prohibir compensar el retraso con exceso de velocidad; exigir conducción defensiva en lluvia y tráfico alto",
                "**Ventana de tiempo**: Reconocer que la ventana 12:30–16:00 ya es inalcanzable; no forzar la operación",
                "**Planificación**: Reprogramar la entrega con una nueva franja/fecha basada en el ETA real (~12h 45min)",
                "**Comunicación con cliente**: Informar de inmediato que no se podrá cumplir la ventana original; acordar nueva franja",
                "**Rutas alternativas**: Evaluar rutas menos congestionadas solo si no aumentan el riesgo; priorizar seguridad",
                "**Capacitación**: Capacitación específica en conducción segura con clima adverso y manejo de estrés por retrasos"
            ],
            "riesgos": {
                "servicio": "Muy alto / crítico",
                "seguridad": "Medio–alto (lluvia + tráfico alto + estilo agresivo + fatiga)",
                "mecanico": "Bajo",
                "global": "Alto (dominante por servicio, con componente de seguridad)"
            }
        },
        "Caso 3: Ruta Larga - Riesgo Mecánico": {
            "descripcion": "Entrega de 400 km con tráfico alto, vehículo con historial de fallas y ventana ajustada",
            "datos": {
                "clima": "Bueno",
                "trafico": "Alto",
                "riesgo_ruta": "Bajo",
                "distancia_km": 400.0,
                "tipo_carga": "Normal",
                "peso_kg": 1000,
                "hora_inicio_entrega": pd.Timestamp("16:30").time(),
                "hora_fin_entrega": pd.Timestamp("19:00").time(),
                "experiencia": 5,
                "antiguedad_camion": 3,
                "fallas_mecanicas": "Si",
                "nivel_combustible": 80.0
            },
            "contexto": {
                "conductor": "C045",
                "frenadas_duras": 18,
                "excesos_velocidad": 5,
                "infracciones": 2,
                "incidentes_carga": 1,
                "accidentes_leves": 1,
                "reclamos": 3,
                "horas_mes": 190,
                "km_mes": 8500,
                "entregas_mes": 95,
                "asistencia_capacitacion": "75%",
                "indice_fatiga": 0.68
            },
            "recomendaciones": [
                "**Conducción**: Conducción defensiva estricta; no intentar recuperar retraso con maniobras agresivas ni exceso de velocidad",
                "**Ventana de tiempo**: Aceptar que la ventana 12:30–16:00 es inviable con el tiempo estimado disponible",
                "**Planificación**: Reprogramar considerando ETA extendido y riesgo mecánico, incluso evaluar moverla a otro vehículo/día",
                "**Comunicación con cliente**: Informar el riesgo mecánico y la necesidad de una reprogramación ordenada y segura",
                "**Rutas alternativas**: Evaluar rutas menos congestionadas y más seguras, sin exigir al vehículo; considerar dividir el trayecto",
                "**Gestión del conductor**: Evitar asignarle varias rutas largas seguidas hasta que mejore su fatiga y estilo de conducción",
                "**Gestión del vehículo**: Realizar revisión mecánica exhaustiva antes de nuevas rutas largas; clasificar como 'en observación'",
                "**Capacitación**: Capacitación en conducción segura en tráfico denso, gestión de fatiga y prevención de fallas"
            ],
            "riesgos": {
                "servicio": "Muy alto / crítico",
                "seguridad": "Medio–alto (tráfico alto + estilo agresivo + fatiga)",
                "mecanico": "Medio–alto (historial de fallas + ruta larga)",
                "global": "Alto (servicio crítico + riesgos de seguridad y mecánicos)"
            }
        }
    }
    
    # ==========================
    # SELECTOR DE MODO
    # ==========================
    st.markdown("### 🎯 Modo de Entrada")
    
    col_modo1, col_modo2 = st.columns(2)
    
    with col_modo1:
        modo_entrada = st.selectbox(
            "Seleccione el modo de entrada",
            options=["📝 Entrada Manual", "📋 Casos Predefinidos"],
            help="Elija entre ingresar datos manualmente o cargar un caso de ejemplo"
        )
    
    # Si se selecciona casos predefinidos
    caso_seleccionado = None
    if modo_entrada == "📋 Casos Predefinidos":
        with col_modo2:
            caso_seleccionado = st.selectbox(
                "Seleccione un caso",
                options=list(casos_predefinidos.keys()),
                help="Casos de ejemplo con diferentes niveles de riesgo"
            )
        
        # Mostrar descripción del caso
        if caso_seleccionado:
            caso = casos_predefinidos[caso_seleccionado]
            st.info(f"📄 **Descripción**: {caso['descripcion']}")
            
            # Mostrar contexto del conductor
            with st.expander("👤 Contexto del Conductor C045"):
                ctx = caso['contexto']
                col_ctx1, col_ctx2, col_ctx3 = st.columns(3)
                
                with col_ctx1:
                    st.metric("Frenadas Duras", ctx['frenadas_duras'])
                    st.metric("Excesos de Velocidad", ctx['excesos_velocidad'])
                    st.metric("Infracciones", ctx['infracciones'])
                
                with col_ctx2:
                    st.metric("Incidentes de Carga", ctx['incidentes_carga'])
                    st.metric("Accidentes Leves", ctx['accidentes_leves'])
                    st.metric("Reclamos", ctx['reclamos'])
                
                with col_ctx3:
                    st.metric("Horas/Mes", f"{ctx['horas_mes']} h")
                    st.metric("Km/Mes", f"{ctx['km_mes']:,} km")
                    st.metric("Índice de Fatiga", ctx['indice_fatiga'], 
                             delta="Elevado" if ctx['indice_fatiga'] > 0.6 else "Normal",
                             delta_color="inverse" if ctx['indice_fatiga'] > 0.6 else "normal")
    
    st.markdown("---")
    
    # ==========================
    # FORMULARIO DE ENTRADA
    # ==========================
    st.markdown("### 📝 Datos del Envío")
    
    # Obtener valores del caso o usar valores por defecto
    if caso_seleccionado:
        valores = casos_predefinidos[caso_seleccionado]['datos']
    else:
        valores = {
            "clima": "Bueno",
            "trafico": "Bajo",
            "riesgo_ruta": "Bajo",
            "distancia_km": 50.0,
            "tipo_carga": "Normal",
            "peso_kg": 1000,
            "hora_inicio_entrega": pd.Timestamp("09:00").time(),
            "hora_fin_entrega": pd.Timestamp("11:00").time(),
            "experiencia": 5,
            "antiguedad_camion": 3,
            "fallas_mecanicas": "No",
            "nivel_combustible": 80.0
        }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🌤️ Condiciones de la Ruta")
        clima = st.selectbox(
            "Clima",
            options=["Bueno", "Lluvia", "Tormenta"],
            index=["Bueno", "Lluvia", "Tormenta"].index(valores["clima"]),
            help="Condiciones meteorológicas esperadas"
        )
        
        trafico = st.selectbox(
            "Nivel de Tráfico",
            options=["Bajo", "Medio", "Alto"],
            index=["Bajo", "Medio", "Alto"].index(valores["trafico"]),
            help="Congestión vehicular esperada"
        )
        
        riesgo_ruta = st.selectbox(
            "Riesgo de Ruta",
            options=["Bajo", "Medio", "Alto"],
            index=["Bajo", "Medio", "Alto"].index(valores["riesgo_ruta"]),
            help="Nivel de riesgo de la ruta (seguridad, condiciones del camino)"
        )
        
        distancia_km = st.number_input(
            "Distancia Total (km)",
            min_value=0.0,
            max_value=1000.0,
            value=float(valores["distancia_km"]),
            step=1.0,
            format="%.2f",
            help="Distancia total de la ruta"
        )
    
    with col2:
        st.markdown("#### 📦 Información de la Carga")
        tipo_carga = st.selectbox(
            "Tipo de Carga",
            options=["Normal", "Fragil", "Peligrosa"],
            index=["Normal", "Fragil", "Peligrosa"].index(valores["tipo_carga"])
        )
        
        peso_kg = st.number_input(
            "Peso de Carga (kg)",
            min_value=0,
            max_value=20000,
            value=int(valores["peso_kg"]),
            step=100,
            help="Peso total de la carga"
        )
        
        st.markdown("##### ⏰ Ventana de Entrega")
        hora_inicio_entrega = st.time_input(
            "Hora inicio de ventana de entrega",
            value=valores["hora_inicio_entrega"],
            help="Primera hora en que se puede entregar"
        )
        
        hora_fin_entrega = st.time_input(
            "Hora fin de ventana de entrega",
            value=valores["hora_fin_entrega"],
            help="Última hora en que se puede entregar"
        )
    
    with col3:
        st.markdown("#### 🚛 Conductor y Vehículo")
        experiencia = st.number_input(
            "Experiencia del Conductor (años)",
            min_value=0,
            max_value=40,
            value=int(valores["experiencia"]),
            help="Años de experiencia en entregas"
        )
        
        antiguedad_camion = st.number_input(
            "Antigüedad del Vehículo (años)",
            min_value=0,
            max_value=30,
            value=int(valores["antiguedad_camion"]),
            help="Edad del vehículo"
        )
        
        fallas_mecanicas = st.selectbox(
            "Historial de Fallas Mecánicas",
            options=["No", "Si"],
            index=["No", "Si"].index(valores["fallas_mecanicas"]),
            help="¿El vehículo ha tenido fallas recientes?"
        )
        
        nivel_combustible = st.number_input(
            "Nivel de Combustible Inicial (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(valores["nivel_combustible"]),
            step=5.0,
            format="%.1f"
        )
    
    st.markdown("---")
    
    # Funciones auxiliares
    VELOCIDAD_BASE = 40
    factores_clima = {"Bueno": 1.0, "Lluvia": 0.8, "Tormenta": 0.5}
    factores_trafico = {"Bajo": 1.0, "Medio": 0.75, "Alto": 0.5}
    
    def calcular_tiempo_estimado(distancia, clima, trafico, experiencia, antiguedad):
        velocidad_efectiva = VELOCIDAD_BASE * factores_clima[clima] * factores_trafico[trafico]
        
        if experiencia < 2:
            velocidad_efectiva *= 0.7
        elif experiencia < 5:
            velocidad_efectiva *= 0.85
        else:
            velocidad_efectiva *= 1.0
        
        if antiguedad > 10:
            velocidad_efectiva *= 0.85
        elif antiguedad > 5:
            velocidad_efectiva *= 0.9
        
        tiempo_viaje_horas = distancia / velocidad_efectiva
        tiempo_minutos = tiempo_viaje_horas * 60
        tiempo_parada = 15
        
        return tiempo_minutos + tiempo_parada
    
    def determinar_horario_salida(hora_actual):
        hora = hora_actual.hour
        
        if 6 <= hora < 12:
            return "Manana"
        elif 12 <= hora < 18:
            return "Tarde"
        else:
            return "Noche"
    
    # Botón de predicción
    if st.button("🔮 Predecir Entrega", type="primary", use_container_width=True):
        
        hora_actual = datetime.now() - timedelta(hours=5)
        horario = determinar_horario_salida(hora_actual)
        
        tiempo_estimado = calcular_tiempo_estimado(
            distancia_km, clima, trafico, experiencia, antiguedad_camion
        )
        
        hora_llegada_estimada = hora_actual + timedelta(minutes=tiempo_estimado)
        hora_inicio_ventana = datetime.combine(hora_actual.date(), hora_inicio_entrega)
        hora_fin_ventana = datetime.combine(hora_actual.date(), hora_fin_entrega)
        
        demora_minutos = (hora_llegada_estimada - hora_inicio_ventana).total_seconds() / 60
        tiempo_real_simulado = tiempo_estimado + (demora_minutos if demora_minutos > 0 else 0)
        
        nueva_entrada = pd.DataFrame([{
            "Clima": clima,
            "TraficoPico": trafico,
            "RiesgoRuta": riesgo_ruta,
            "Distancia_km": distancia_km,
            "TiempoEstimado_min": tiempo_estimado,
            "TiempoReal_min": tiempo_real_simulado,
            "Demora_min": demora_minutos,
            "TipoCarga": tipo_carga,
            "Peso_kg": peso_kg,
            "ExperienciaConductor_anios": experiencia,
            "AntiguedadCamion_anios": antiguedad_camion,
            "FallasMecanicas": fallas_mecanicas,
            "NivelCombustible_pct": nivel_combustible,
            "HorarioSalida": horario,
        }])
        
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
        
        st.info(f"🎯 **Ventana de entrega:** {hora_inicio_entrega.strftime('%H:%M')} - {hora_fin_entrega.strftime('%H:%M')}")
        
        if demora_minutos <= 0:
            st.success("✅ Llegará a tiempo")
        else:
            st.error("❌ Llegará tarde")
        
        # Mostrar recomendaciones si es un caso predefinido
        if caso_seleccionado:
            caso = casos_predefinidos[caso_seleccionado]
            
            st.markdown("---")
            st.subheader("💡 Recomendaciones y Plan de Acción")
            
            # Tabla de riesgos
            st.markdown("#### ⚠️ Evaluación de Riesgos")
            riesgos = caso['riesgos']
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.metric("Riesgo de Servicio", riesgos['servicio'])
            with col_r2:
                st.metric("Riesgo de Seguridad", riesgos['seguridad'])
            with col_r3:
                st.metric("Riesgo Mecánico", riesgos['mecanico'])
            with col_r4:
                st.metric("Riesgo Global", riesgos['global'])
            
            # Recomendaciones
            st.markdown("#### 📋 Acciones Recomendadas")
            for recomendacion in caso['recomendaciones']:
                st.markdown(f"- {recomendacion}")

# ==========================
# MÓDULO 2: CLUSTERING + PCA
# ==========================
else:
    st.write("Análisis de comportamiento de conductores mediante clustering y reducción dimensional con PCA.")
    
    st.markdown("### 👤 Ingrese los datos del conductor")
    
    # Definir límites máximos
    limites = {
        "frenadas_duras": 50,
        "excesos_velocidad": 30,
        "incidentes_carga": 20,
        "infracciones": 15,
        "horas_manejo_mes": 220,
        "km_mes": 8000,
        "entregas_mes": 300,
        "reclamos_clientes": 25,
        "accidentes_leves": 10,
        "asistencia_capacitaciones": 12,
        "indice_fatiga": 10
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🚗 Comportamiento de Manejo")
        frenadas_duras = st.number_input(
            "Frenadas Duras",
            min_value=0,
            max_value=limites["frenadas_duras"],
            value=5,
            help="Número de frenadas bruscas en el período"
        )
        
        excesos_velocidad = st.number_input(
            "Excesos de Velocidad",
            min_value=0,
            max_value=limites["excesos_velocidad"],
            value=2,
            help="Número de veces que excedió el límite de velocidad"
        )
        
        incidentes_carga = st.number_input(
            "Incidentes con Carga",
            min_value=0,
            max_value=limites["incidentes_carga"],
            value=1,
            help="Problemas con la carga (daños, pérdidas)"
        )
        
        infracciones = st.number_input(
            "Infracciones de Tránsito",
            min_value=0,
            max_value=limites["infracciones"],
            value=0,
            help="Multas o infracciones registradas"
        )
    
    with col2:
        st.markdown("#### 📊 Métricas Operacionales")
        horas_manejo_mes = st.number_input(
            "Horas de Manejo al Mes",
            min_value=0,
            max_value=limites["horas_manejo_mes"],
            value=160,
            help="Total de horas conduciendo en el mes"
        )
        
        km_mes = st.number_input(
            "Kilómetros al Mes",
            min_value=0,
            max_value=limites["km_mes"],
            value=4000,
            step=100,
            help="Distancia total recorrida en el mes"
        )
        
        entregas_mes = st.number_input(
            "Entregas al Mes",
            min_value=0,
            max_value=limites["entregas_mes"],
            value=120,
            help="Número de entregas completadas"
        )
        
        reclamos_clientes = st.number_input(
            "Reclamos de Clientes",
            min_value=0,
            max_value=limites["reclamos_clientes"],
            value=2,
            help="Quejas o reclamos recibidos"
        )
    
    with col3:
        st.markdown("#### 🏥 Seguridad y Capacitación")
        accidentes_leves = st.number_input(
            "Accidentes Leves",
            min_value=0,
            max_value=limites["accidentes_leves"],
            value=0,
            help="Accidentes menores sin heridos graves"
        )
        
        asistencia_capacitaciones = st.number_input(
            "Asistencia a Capacitaciones",
            min_value=0,
            max_value=limites["asistencia_capacitaciones"],
            value=8,
            help="Número de capacitaciones completadas en el año"
        )
        
        indice_fatiga = st.slider(
            "Índice de Fatiga",
            min_value=0,
            max_value=limites["indice_fatiga"],
            value=4,
            help="Nivel de fatiga promedio (0=ninguna, 10=extrema)"
        )
    
    st.markdown("---")
    
    # Botón de análisis
    if st.button("📈 Analizar Conductor", type="primary", use_container_width=True):
    
        # Crear DataFrame con los datos
        datos_conductor = pd.DataFrame([{
            "frenadas_duras": frenadas_duras,
            "excesos_velocidad": excesos_velocidad,
            "incidentes_carga": incidentes_carga,
            "infracciones": infracciones,
            "horas_manejo_mes": horas_manejo_mes,
            "km_mes": km_mes,
            "entregas_mes": entregas_mes,
            "reclamos_clientes": reclamos_clientes,
            "accidentes_leves": accidentes_leves,
            "asistencia_capacitaciones": asistencia_capacitaciones,
            "indice_fatiga": indice_fatiga
        }])
        
        # Generar datos sintéticos para comparación
        np.random.seed(42)
        n_conductores = 200
        
        datos_simulados = pd.DataFrame({
            "frenadas_duras": np.random.poisson(8, n_conductores),
            "excesos_velocidad": np.random.poisson(4, n_conductores),
            "incidentes_carga": np.random.poisson(2, n_conductores),
            "infracciones": np.random.poisson(1, n_conductores),
            "horas_manejo_mes": np.random.normal(160, 30, n_conductores).clip(80, 220),
            "km_mes": np.random.normal(4500, 1000, n_conductores).clip(1000, 8000),
            "entregas_mes": np.random.normal(120, 40, n_conductores).clip(30, 300),
            "reclamos_clientes": np.random.poisson(3, n_conductores),
            "accidentes_leves": np.random.poisson(1, n_conductores),
            "asistencia_capacitaciones": np.random.normal(6, 2, n_conductores).clip(0, 12),
            "indice_fatiga": np.random.normal(5, 2, n_conductores).clip(0, 10)
        })
        
        # Combinar datos
        datos_completos = pd.concat([datos_simulados, datos_conductor], ignore_index=True)
        
        # ================================
        # PCA POR BLOQUES (3 SCORES)
        # ================================
        cols_riesgo = ["frenadas_duras", "excesos_velocidad", "incidentes_carga", "infracciones"]
        cols_experiencia = ["horas_manejo_mes", "km_mes", "entregas_mes"]
        cols_seguridad = ["reclamos_clientes", "accidentes_leves", "asistencia_capacitaciones", "indice_fatiga"]
        
        def escalar_0_100(valor, todos):
            vmin = todos.min()
            vmax = todos.max()
            if vmax == vmin:
                return 50.0
            return 100 * (valor - vmin) / (vmax - vmin)
        
        # --- PCA 1: Score de Riesgo ---
        scaler_riesgo = StandardScaler()
        riesgo_scaled = scaler_riesgo.fit_transform(datos_completos[cols_riesgo])
        pca_riesgo = PCA(n_components=1)
        riesgo_pc1 = pca_riesgo.fit_transform(riesgo_scaled).flatten()
        
        # CORRECCIÓN: Verificar dirección del PCA de riesgo
        # Los loadings indican cómo cada variable contribuye al componente
        loadings_riesgo = pca_riesgo.components_[0]
        
        # Calcular el promedio de los loadings (todas las variables de riesgo deben contribuir positivamente)
        # Si la mayoría son negativos, invertimos la dirección
        if np.mean(loadings_riesgo) < 0:
            riesgo_pc1 = -riesgo_pc1
        
        score_riesgo = escalar_0_100(riesgo_pc1[-1], riesgo_pc1)
        
        if score_riesgo < 33:
            nivel_riesgo = "Bajo"
        elif score_riesgo < 66:
            nivel_riesgo = "Medio"
        else:
            nivel_riesgo = "Alto"
        
        # --- PCA 2: Experticia ---
        scaler_exp = StandardScaler()
        exp_scaled = scaler_exp.fit_transform(datos_completos[cols_experiencia])
        pca_exp = PCA(n_components=1)
        exp_pc1 = pca_exp.fit_transform(exp_scaled).flatten()
        
        # Verificar dirección (más horas/km/entregas = más experticia)
        loadings_exp = pca_exp.components_[0]
        if np.mean(loadings_exp) < 0:
            exp_pc1 = -exp_pc1
        
        score_exp = escalar_0_100(exp_pc1[-1], exp_pc1)
        
        if score_exp < 33:
            nivel_exp = "Junior"
        elif score_exp < 66:
            nivel_exp = "Intermedio"
        else:
            nivel_exp = "Senior"
        
        # --- PCA 3: Seguridad / Fatiga ---
        seg_df = datos_completos[cols_seguridad].copy()
        seg_df["asistencia_capacitaciones"] = -seg_df["asistencia_capacitaciones"]
        
        scaler_seg = StandardScaler()
        seg_scaled = scaler_seg.fit_transform(seg_df)
        pca_seg = PCA(n_components=1)
        seg_pc1 = pca_seg.fit_transform(seg_scaled).flatten()
        
        score_seg = escalar_0_100(seg_pc1[-1], seg_pc1)
        
        if score_seg < 33:
            nivel_seg = "Buena seguridad / baja fatiga"
        elif score_seg < 66:
            nivel_seg = "Vigilancia necesaria"
        else:
            nivel_seg = "Crítico (alto riesgo / fatiga)"
        
        # ================================
        # CLUSTERING BASADO EN RIESGO Y EXPERTICIA
        # ================================
        # Crear matriz de features para clustering (solo Riesgo y Experticia)
        features_clustering = np.column_stack([riesgo_pc1, exp_pc1])
        
        # Aplicar K-Means sobre estos dos componentes
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_clustering)
        
        # Identificar cluster del conductor
        cluster_conductor = clusters[-1]
        
        # ASIGNAR NOMBRES BASADOS EN CENTROIDES
        centroides = kmeans.cluster_centers_
        nombres_clusters = {}
        
        for i, centroide in enumerate(centroides):
            riesgo_centroide = centroide[0]
            exp_centroide = centroide[1]
            
            # Clasificar según cuadrante
            if riesgo_centroide < np.median(riesgo_pc1) and exp_centroide > np.median(exp_pc1):
                nombres_clusters[i] = "🟢 MAESTRO_IDEAL"
            elif riesgo_centroide < np.median(riesgo_pc1) and exp_centroide < np.median(exp_pc1):
                nombres_clusters[i] = "🟡 NOVATO_SEGURO"
            elif riesgo_centroide > np.median(riesgo_pc1) and exp_centroide > np.median(exp_pc1):
                nombres_clusters[i] = "🟠 EXPERTO_RIESGOSO"
            else:
                nombres_clusters[i] = "🔴 NOVATO_RIESGOSO"
        
        # ================================
        # PCA 2D GLOBAL PARA VISUALIZACIÓN
        # ================================
        scaler_global = StandardScaler()
        datos_normalizados_global = scaler_global.fit_transform(datos_completos)
        pca_global = PCA(n_components=2)
        datos_pca_global = pca_global.fit_transform(datos_normalizados_global)
        
        df_viz = pd.DataFrame({
            "PC1": datos_pca_global[:, 0],
            "PC2": datos_pca_global[:, 1],
            "Cluster": clusters,
            "Tipo": ["Otros Conductores"] * n_conductores + ["Conductor Actual"]
        })
        
        # ================================
        # MOSTRAR RESULTADOS
        # ================================
        st.subheader("📊 Resultados del Análisis")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("🎯 Cluster Asignado", nombres_clusters[cluster_conductor])
        
        with col_m2:
            st.metric("⚠️ Nivel de Riesgo", nivel_riesgo)
        
        with col_m3:
            st.metric("👨‍💼 Nivel de Experticia", nivel_exp)
        
        st.markdown("---")
        
        # Mostrar los 3 PCA como scores
        st.markdown("### 🎛 Scores PCA por dimensión")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric(
                "Score de Riesgo (0-100)",
                f"{score_riesgo:.1f}%",
                help="Basado en frenadas duras, excesos de velocidad, incidentes con carga e infracciones. Mayor score = Mayor riesgo"
            )
            st.write(f"Nivel de riesgo: **{nivel_riesgo}**")
            
            # Mostrar loadings para transparencia
            with st.expander("📊 Ver contribución de variables"):
                for var, loading in zip(cols_riesgo, pca_riesgo.components_[0]):
                    st.write(f"- {var}: {loading:.3f}")
        
        with c2:
            st.metric(
                "Score de Experticia (0-100)",
                f"{score_exp:.1f}%",
                help="Basado en horas de manejo, km recorridos y entregas realizadas."
            )
            st.write(f"Nivel de experticia: **{nivel_exp}**")
            
            with st.expander("📊 Ver contribución de variables"):
                for var, loading in zip(cols_experiencia, pca_exp.components_[0]):
                    st.write(f"- {var}: {loading:.3f}")
        
        with c3:
            st.metric(
                "Índice Seguridad / Fatiga (0-100)",
                f"{score_seg:.1f}%",
                help="Basado en reclamos, accidentes, capacitaciones y nivel de fatiga."
            )
            st.write(f"Situación: **{nivel_seg}**")
        
        st.markdown("---")
        
        # Análisis de características del cluster
        st.markdown("### 🔍 Características del Cluster")
        
        mask_cluster = clusters[:-1] == cluster_conductor
        if mask_cluster.sum() > 0:
            stats_cluster = datos_simulados[mask_cluster].mean()
            
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("#### Promedios del Cluster")
                st.write(f"**Frenadas Duras:** {stats_cluster['frenadas_duras']:.1f}")
                st.write(f"**Excesos Velocidad:** {stats_cluster['excesos_velocidad']:.1f}")
                st.write(f"**Incidentes Carga:** {stats_cluster['incidentes_carga']:.1f}")
                st.write(f"**Infracciones:** {stats_cluster['infracciones']:.1f}")
                st.write(f"**Accidentes Leves:** {stats_cluster['accidentes_leves']:.1f}")
            
            with col_s2:
                st.markdown("#### Tus Valores")
                st.write(f"**Frenadas Duras:** {frenadas_duras}")
                st.write(f"**Excesos Velocidad:** {excesos_velocidad}")
                st.write(f"**Incidentes Carga:** {incidentes_carga}")
                st.write(f"**Infracciones:** {infracciones}")
                st.write(f"**Accidentes Leves:** {accidentes_leves}")
        
        # Recomendaciones según cluster
        st.markdown("### 💡 Recomendaciones")
        
        if "MAESTRO_IDEAL" in nombres_clusters[cluster_conductor]:
            st.success("¡Excelente desempeño! Mantén tus buenos hábitos de conducción y seguridad.")
        elif "NOVATO_SEGURO" in nombres_clusters[cluster_conductor]:
            st.info("Buen perfil de seguridad. Aumenta tu experiencia y productividad para avanzar.")
        elif "EXPERTO_RIESGOSO" in nombres_clusters[cluster_conductor]:
            st.warning("Alta experiencia pero con comportamientos riesgosos. Reduce infracciones y mejora hábitos de conducción.")
        else:
            st.error("Requiere atención inmediata. Necesitas mejorar tanto en experiencia como en seguridad.")

st.markdown("---")
st.caption("🔧 Sistema de Análisis de Entregas v3.0 | Hora actual: " + datetime.now().strftime("%H:%M:%S"))