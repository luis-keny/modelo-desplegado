import pandas as pd
import joblib

# Cargar el modelo entrenado
pipe = joblib.load("modelo_entregas_mlp.pkl")

# Crear nueva entrada con los datos de predicción
nueva_entrada = pd.DataFrame([{
    "Clima": "Lluvia",
    "TraficoPico": "Bajo",
    "RiesgoRuta": "Medio",
    "Distancia_km": 291.94,
    "TiempoEstimado_min": 291.9,
    "TiempoReal_min": 327.5,
    "Demora_min": 35.6,
    "TipoCarga": "Fragil",
    "Peso_kg": 4013,
    "ExperienciaConductor_anios": 5,
    "AntiguedadCamion_anios": 6,
    "FallasMecanicas": "No",
    "NivelCombustible_pct": 65.6,
    "HorarioSalida": "Noche",
}])

# Realizar predicción
pred = pipe.predict(nueva_entrada)        # 0 o 1
prob = pipe.predict_proba(nueva_entrada)  # probabilidad

# Mostrar resultados
print(f"Predicción: {pred[0]}")
print(f"Probabilidad: {prob[0]}")
