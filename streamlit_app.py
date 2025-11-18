# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Configuración de la página
st.set_page_config(page_title="Predicción de Entregas", page_icon="🚚", layout="wide")

# 1. Cargar el scaler
SCALER_PATH = Path("artefactos") / "scaler.pkl"

try:
    scaler = joblib.load(SCALER_PATH)
    st.success("✅ Scaler cargado exitosamente")
except Exception as e:
    st.error(f"❌ Error al cargar el scaler: {e}")
    st.stop()

st.title("🚚 Escalador de Datos para Entregas")
st.write("Transforma los datos de entrada usando el StandardScaler entrenado")

# Tabs
tab1, tab2 = st.tabs(["📝 Escalado de Datos", "ℹ️ Información del Scaler"])

# --- TAB 1: Entrada y escalado ---
with tab1:
    st.subheader("Ingrese los datos del envío para escalar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌤️ Condiciones Ambientales")
        clima = st.selectbox(
            "Clima",
            options=["Bueno", "Lluvia", "Tormenta"],
            help="Condiciones meteorológicas durante la entrega"
        )
        trafico = st.selectbox(
            "Nivel de Tráfico",
            options=["Bajo", "Medio", "Alto"],
            help="Congestión vehicular en la ruta"
        )
        horario = st.selectbox(
            "Horario de Salida",
            options=["Manana", "Tarde", "Noche"]
        )
    
    with col2:
        st.markdown("### 📦 Información de Carga")
        tipo_carga = st.selectbox(
            "Tipo de Carga",
            options=["Normal", "Fragil", "Peligrosa"]
        )
        peso_kg = st.number_input(
            "Peso de Carga (kg)",
            min_value=0,
            max_value=20000,
            value=8000,
            step=100
        )
        distancia_km = st.number_input(
            "Distancia (km)",
            min_value=0.0,
            max_value=400.0,
            value=150.0,
            step=1.0,
            format="%.2f"
        )
        tiempo_estimado = st.number_input(
            "Tiempo Estimado (min)",
            min_value=0.0,
            max_value=600.0,
            value=180.0,
            step=1.0,
            format="%.1f"
        )
        tiempo_real = st.number_input(
            "Tiempo Real (min)",
            min_value=0.0,
            max_value=800.0,
            value=200.0,
            step=1.0,
            format="%.1f"
        )
        demora = st.number_input(
            "Demora (min)",
            min_value=-50.0,
            max_value=200.0,
            value=20.0,
            step=1.0,
            format="%.1f"
        )
    
    with col3:
        st.markdown("### 🚛 Información del Vehículo")
        experiencia = st.number_input(
            "Experiencia del Conductor (años)",
            min_value=0,
            max_value=20,
            value=5
        )
        antiguedad_camion = st.number_input(
            "Antigüedad del Camión (años)",
            min_value=0,
            max_value=15,
            value=5
        )
        fallas_mecanicas = st.selectbox(
            "Fallas Mecánicas",
            options=["No", "Si"]
        )
        nivel_combustible = st.number_input(
            "Nivel de Combustible (%)",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=0.1,
            format="%.1f"
        )
    
    st.markdown("---")
    
    # Botón para escalar
    if st.button("🔄 Escalar Datos", type="primary", use_container_width=True):
        map_trafico = {
            "Bajo": 0.0,
            "Medio": 1.0,
            "Alto": 2.0
        }
        trafico_val = map_trafico[trafico]

        fallas_mecanicas_val = 1 if fallas_mecanicas == 'Si'else 0
        # One-hot para Clima
        clima_bueno    = 1 if clima == "Bueno"    else 0
        clima_lluvia   = 1 if clima == "Lluvia"   else 0
        clima_tormenta = 1 if clima == "Tormenta" else 0

        # One-hot para TipoCarga
        tipo_fragil     = 1 if tipo_carga == "Fragil"    else 0
        tipo_normal     = 1 if tipo_carga == "Normal"    else 0
        tipo_peligrosa  = 1 if tipo_carga == "Peligrosa" else 0

        # One-hot para HorarioSalida
        hor_manana = 1 if horario == "Manana" else 0
        hor_noche  = 1 if horario == "Noche"  else 0
        hor_tarde  = 1 if horario == "Tarde"  else 0

        # DataFrame de entrada con 19 features (sin RiesgoRuta)
        datos_entrada = pd.DataFrame({
            # numéricas / ordinales originales
            'TraficoPico': [trafico_val],
            'Distancia_km': [distancia_km],
            'TiempoEstimado_min': [tiempo_estimado],
            'TiempoReal_min': [tiempo_real],
            'Demora_min': [demora], # si ya no la usas directo, puedes omitirla del modelo
            'Peso_kg': [peso_kg],
            'ExperienciaConductor_anios': [experiencia],
            'AntiguedadCamion_anios': [antiguedad_camion],
            'FallasMecanicas': [fallas_mecanicas_val],
            'NivelCombustible_pct': [nivel_combustible],

            # one-hot clima
            'Clima_Bueno':   [clima_bueno],
            'Clima_Lluvia':  [clima_lluvia],
            'Clima_Tormenta':[clima_tormenta],

            # one-hot tipo de carga
            'TipoCarga_Fragil':    [tipo_fragil],
            'TipoCarga_Normal':    [tipo_normal],
            'TipoCarga_Peligrosa': [tipo_peligrosa],

            # one-hot horario de salida
            'HorarioSalida_Manana': [hor_manana],
            'HorarioSalida_Noche':  [hor_noche],
            'HorarioSalida_Tarde':  [hor_tarde],
        })
        
        st.subheader("📊 Datos Originales")
        st.dataframe(datos_entrada, use_container_width=True)
        
        # Aplicar One-Hot Encoding a las variables categóricas
        datos_encoded = datos_entrada.copy()
        
        st.subheader("🔢 Datos después de One-Hot Encoding")
        st.dataframe(datos_encoded, use_container_width=True)
        st.info(f"📏 Forma de los datos: {datos_encoded.shape} (filas, columnas)")
        
        # Aplicar el scaler
        try:
            datos_escalados = scaler.transform(datos_encoded)
            
            st.subheader("✨ Datos Escalados (Normalizados)")
            
            # Convertir a DataFrame para mejor visualización
            df_escalados = pd.DataFrame(
                datos_escalados, 
                columns=datos_encoded.columns
            )
            st.dataframe(df_escalados, use_container_width=True)
            
            # Mostrar estadísticas
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("📉 Valor Mínimo", f"{datos_escalados.min():.4f}")
            with col_stats2:
                st.metric("📊 Valor Promedio", f"{datos_escalados.mean():.4f}")
            with col_stats3:
                st.metric("📈 Valor Máximo", f"{datos_escalados.max():.4f}")
            
            # Opción para descargar
            st.markdown("---")
            st.subheader("💾 Exportar Datos")
            
            csv_encoded = datos_encoded.to_csv(index=False)
            csv_escalados = df_escalados.to_csv(index=False)
            
            col_down1, col_down2 = st.columns(2)
            
            with col_down1:
                st.download_button(
                    label="📥 Descargar Datos Encoded (CSV)",
                    data=csv_encoded,
                    file_name="datos_encoded.csv",
                    mime="text/csv"
                )
            
            with col_down2:
                st.download_button(
                    label="📥 Descargar Datos Escalados (CSV)",
                    data=csv_escalados,
                    file_name="datos_escalados.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error al escalar los datos: {e}")
            st.info("💡 Verifica que las columnas coincidan con las que el scaler espera.")

# --- TAB 2: Información del Scaler ---
with tab2:
    st.subheader("ℹ️ Información del StandardScaler")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### 📋 Características del Scaler")
        
        if hasattr(scaler, 'n_features_in_'):
            st.metric("Número de características", scaler.n_features_in_)
        
        if hasattr(scaler, 'feature_names_in_'):
            st.write("**Características esperadas:**")
            st.write(list(scaler.feature_names_in_))
        
        if hasattr(scaler, 'mean_'):
            st.write(f"**Media calculada:** {len(scaler.mean_)} valores")
        
        if hasattr(scaler, 'scale_'):
            st.write(f"**Escala calculada:** {len(scaler.scale_)} valores")
    
    with col_info2:
        st.markdown("### 🔍 ¿Qué hace el StandardScaler?")
        st.markdown("""
        El **StandardScaler** transforma los datos para que tengan:
        - **Media = 0**
        - **Desviación estándar = 1**
        
        **Fórmula:**
        ```
        X_scaled = (X - μ) / σ
        ```
        Donde:
        - `X` = valor original
        - `μ` = media del conjunto de entrenamiento
        - `σ` = desviación estándar del conjunto de entrenamiento
        
        Esto es importante para que las redes neuronales funcionen correctamente.
        """)
    
    st.markdown("---")
    
    # Mostrar estadísticas del scaler si están disponibles
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        st.subheader("📊 Estadísticas del Scaler")
        
        stats_df = pd.DataFrame({
            'Media (μ)': scaler.mean_,
            'Desviación Estándar (σ)': scaler.scale_
        })
        
        if hasattr(scaler, 'feature_names_in_'):
            stats_df.index = scaler.feature_names_in_
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Visualización
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Media (μ)',
            x=list(range(len(scaler.mean_))),
            y=scaler.mean_,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Desviación Estándar (σ)',
            x=list(range(len(scaler.scale_))),
            y=scaler.scale_,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Parámetros del Scaler por Característica',
            xaxis_title='Índice de Característica',
            yaxis_title='Valor',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔧 Sistema de Escalado de Datos v1.0 | StandardScaler</p>
</div>
""", unsafe_allow_html=True)