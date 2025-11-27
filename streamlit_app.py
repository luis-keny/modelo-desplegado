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
    
    # Formulario de entrada
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
        
        # Generar datos sintéticos para comparación (simulación de otros conductores)
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
        
        # Normalizar datos
        scaler = StandardScaler()
        datos_normalizados = scaler.fit_transform(datos_completos)
        
        # Aplicar PCA
        pca = PCA(n_components=2)
        datos_pca = pca.fit_transform(datos_normalizados)
        
        # Aplicar K-Means
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(datos_normalizados)
        
        # Identificar el cluster del conductor actual
        cluster_conductor = clusters[-1]
        
        # Crear DataFrame para visualización
        df_viz = pd.DataFrame({
            "PC1": datos_pca[:, 0],
            "PC2": datos_pca[:, 1],
            "Cluster": clusters,
            "Tipo": ["Otros Conductores"] * n_conductores + ["Conductor Actual"]
        })
        
        # Nombres de clusters
        nombres_clusters = {
            0: "🟢 Conductores Ejemplares",
            1: "🟡 Conductores Promedio",
            2: "🟠 Conductores en Desarrollo",
            3: "🔴 Conductores de Alto Riesgo"
        }
        
        st.subheader("📊 Resultados del Análisis")
        
        # Métricas principales
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("🎯 Cluster Asignado", nombres_clusters.get(cluster_conductor, f"Cluster {cluster_conductor}"))
        
        with col_m2:
            varianza_explicada = sum(pca.explained_variance_ratio_[:2]) * 100
            st.metric("📉 Varianza Explicada (PCA)", f"{varianza_explicada:.1f}%")
        
        with col_m3:
            conductores_mismo_cluster = sum(clusters[:-1] == cluster_conductor)
            st.metric("👥 Conductores en tu Cluster", conductores_mismo_cluster)
        
        st.markdown("---")
        
        # Visualización PCA + Clustering
        fig = go.Figure()
        
        # Plotear otros conductores
        for cluster_id in range(n_clusters):
            mask = (df_viz["Tipo"] == "Otros Conductores") & (df_viz["Cluster"] == cluster_id)
            fig.add_trace(go.Scatter(
                x=df_viz[mask]["PC1"],
                y=df_viz[mask]["PC2"],
                mode="markers",
                name=nombres_clusters.get(cluster_id, f"Cluster {cluster_id}"),
                marker=dict(size=8, opacity=0.6),
                showlegend=True
            ))
        
        # Plotear conductor actual
        conductor_mask = df_viz["Tipo"] == "Conductor Actual"
        fig.add_trace(go.Scatter(
            x=df_viz[conductor_mask]["PC1"],
            y=df_viz[conductor_mask]["PC2"],
            mode="markers",
            name="TÚ",
            marker=dict(size=20, symbol="star", color="gold", line=dict(width=2, color="black")),
            showlegend=True
        ))
        
        # Plotear centroides
        centroides_pca = pca.transform(scaler.transform(kmeans.cluster_centers_))
        fig.add_trace(go.Scatter(
            x=centroides_pca[:, 0],
            y=centroides_pca[:, 1],
            mode="markers",
            name="Centroides",
            marker=dict(size=15, symbol="x", color="black", line=dict(width=2)),
            showlegend=True
        ))
        
        fig.update_layout(
            title="Análisis PCA + K-Means: Posición del Conductor",
            xaxis_title=f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)",
            yaxis_title=f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)",
            height=600,
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de características principales
        st.markdown("### 🔍 Características del Cluster")
        
        # Calcular estadísticas del cluster
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
        
        if cluster_conductor == 0:
            st.success("¡Excelente desempeño! Mantén tus buenos hábitos de conducción y seguridad.")
        elif cluster_conductor == 1:
            st.info("Rendimiento aceptable. Considera mejorar en áreas de seguridad y capacitación.")
        elif cluster_conductor == 2:
            st.warning("Hay espacio para mejorar. Enfócate en reducir incidentes y aumentar capacitación.")
        else:
            st.error("Requiere atención inmediata. Es necesario mejorar hábitos de conducción y seguridad.")

st.markdown("---")
st.caption("🔧 Sistema de Análisis de Entregas v3.0 | Hora actual: " + datetime.now().strftime("%H:%M:%S"))