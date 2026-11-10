import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Predictor de Rendimiento y Comportamiento Académico")
st.write("Proyecto 3 - Métodos Estadísticos: Modelos de Ridge Regression y Random Forest")

@st.cache_data
def cargar_datos():
    # Leer el archivo con codificación para evitar errores de símbolos
    df = pd.read_csv('datos_estudiantes.csv', encoding='latin1')
    return df

try:
    df = cargar_datos()
    st.success("¡Datos cargados correctamente desde GitHub!")

    # Limpiamos las columnas que no son numéricas para la predicción (como fechas, correos, nombres)
    # Seleccionamos únicamente columnas numéricas o convertimos las que tengan datos válidos
    df_num = df.select_dtypes(include=[np.number]).dropna(how='all', axis=1)

    # Si hay suficientes columnas numéricas
    if df_num.shape[1] >= 2:
        # Rellenamos valores vacíos con la media de cada columna para evitar errores de NaN
        df_num = df_num.fillna(df_num.mean())

        # Definimos X (variables predictoras/comportamientos) y Y (calificación a predecir)
        # La última columna numérica será Y (calificación), y el resto serán X (hábitos/preguntas)
        X = df_num.iloc[:, :-1]
        Y = df_num.iloc[:, -1]

        if len(X.columns) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            # 1. Entrenar Ridge Regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            ridge_pred = ridge.predict(X_test)

            # 2. Entrenar Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            st.subheader("Evaluación de los Modelos de Predicción")
            st.write(f"**Ridge Regression - R2 Score:** {r2_score(y_test, ridge_pred):.2f}")
            st.write(f"**Random Forest - R2 Score:** {r2_score(y_test, rf_pred):.2f}")

            st.markdown("---")
            st.subheader("Simulador de Comportamiento y Predicción")
            st.write("Modifica los valores de los hábitos de estudio para predecir el resultado:")

            # Controles interactivos para cada variable predictora
            valores_usuario = []
            for col in X.columns:
                val_min = float(X[col].min())
                val_max = float(X[col].max())
                val_mean = float(X[col].mean())
                val = st.slider(f"Variable / Pregunta: {col}", val_min, val_max, val_mean)
                valores_usuario.append(val)

            if st.button("Predecir Comportamiento / Calificación"):
                entrada = np.array([valores_usuario])
                res_ridge = ridge.predict(entrada)[0]
                res_rf = rf.predict(entrada)[0]

                st.success(f"📌 **Predicción estimada (Ridge Regression):** {res_ridge:.2f}")
                st.success(f"🌲 **Predicción estimada (Random Forest):** {res_rf:.2f}")
        else:
            st.error("No hay suficientes variables numéricas de comportamiento para entrenar el modelo.")
    else:
        st.error("El archivo CSV no contiene suficientes columnas numéricas legibles.")

except Exception as e:
    st.error(f"Error en el procesamiento: {e}")
