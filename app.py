import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Predictor de Rendimiento Académico (UAP)")
st.write("Proyecto 3 - Métodos Estadísticos: Modelos de Ridge Regression y Random Forest")

# 1. Cargar datos (asegúrate de tener tu archivo CSV en la misma carpeta)
# Si tu archivo se llama diferente, cambia 'datos_estudiantes.csv' por el nombre real.
@st.cache_data
def cargar_datos():
    # Simulación o lectura real de tu CSV con los 59 estudiantes
    df = pd.read_csv('datos_estudiantes.csv')
    return df

try:
    df = cargar_datos()
    
    # Definir variables X (independientes) y Y (calificación del periodo anterior)
    # Ajusta los nombres de tus columnas según tu archivo CSV
    X = df[['study_hours', 'work_hours', 'sleep_hours', 'attendance', 'assignments', 'current_grade']]
    Y = df['past_grade']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Entrenar modelo Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    # Entrenar modelo Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    st.subheader("Evaluación de los Modelos en el conjunto de prueba")
    st.write(f"**Ridge Regression - MSE:** {mean_squared_error(y_test, ridge_pred):.2f} | **R2 Score:** {r2_score(y_test, ridge_pred):.2f}")
    st.write(f"**Random Forest - MSE:** {mean_squared_error(y_test, rf_pred):.2f} | **R2 Score:** {r2_score(y_test, rf_pred):.2f}")

    st.markdown("---")
    st.subheader("Realiza una Predicción Individual")

    # Controles deslizantes en la app de Streamlit para ingresar nuevos datos
    study_h = st.slider("Horas de estudio por semana", 0, 20, 5)
    work_h = st.slider("Horas de trabajo al día", 0, 10, 0)
    sleep_h = st.slider("Horas de sueño por noche", 0, 12, 7)
    attendance_d = st.slider("Días de asistencia por semana", 1, 6, 5)
    assignments_f = st.slider("Frecuencia de tareas (1 al 5)", 1, 5, 4)
    current_g = st.number_input("Calificación del periodo actual", 0.0, 10.0, 8.5)

    # Botón para predecir
    if st.button("Predecir Calificación Anterior"):
        entrada_usuario = np.array([[study_h, work_h, sleep_h, attendance_d, assignments_f, current_g]])
        
        res_ridge = ridge.predict(entrada_usuario)[0]
        res_rf = rf.predict(entrada_usuario)[0]

        st.success(f"📌 **Predicción con Ridge Regression:** {res_ridge:.2f}")
        st.success(f"🌲 **Predicción con Random Forest:** {res_rf:.2f}")

except Exception as e:
    st.error("Por favor, sube tu archivo de datos 'datos_estudiantes.csv' en el repositorio de GitHub para que la app pueda leerlo.")