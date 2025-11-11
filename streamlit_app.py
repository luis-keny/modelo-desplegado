# mi_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Ruta al modelo dentro de artefactos/
MODEL_PATH = Path("artefactos") / "modelo_pima.pkl"
modelo = joblib.load(MODEL_PATH)

st.title("🤖 Predicción de Diabetes (Pima Dataset)")
st.write("Ingrese los valores clínicos para predecir si la paciente probablemente tiene diabetes.")

data = {
    'npreg': st.slider("Número de embarazos", 0, 20, 2),
    'glu':   st.slider("Nivel de glucosa (mg/dl)", 50, 200, 100),
    'bp':    st.slider("Presión arterial (mmHg)", 40, 130, 70),
    'skin':  st.slider("Espesor del pliegue cutáneo (mm)", 7, 100, 20),
    'bmi':   st.slider("IMC", 10.0, 50.0, 25.0),
    'ped':   st.slider("Pedigree de diabetes", 0.0, 2.5, 0.5),
    'age':   st.slider("Edad (años)", 18, 90, 35)
}

if st.button("Predecir"):
    entrada = pd.DataFrame([data])
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]
    resultado = "Diabética" if pred == 1 else "No diabética"
    st.write(f"Resultado: **{resultado}**")
    st.write(f"Probabilidad estimada: **{prob:.2f}**")
