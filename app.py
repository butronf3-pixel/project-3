import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Predictor de Rendimiento Académico (UAP)")
st.write("Proyecto 3 - Métodos Estadísticos: Modelos de Ridge Regression y Random Forest")

@st.cache_data
def cargar_datos():
    # Añadimos encoding='latin1' para evitar el error de caracteres especiales
    df = pd.read_csv('datos_estudiantes.csv', encoding='latin1')
    return df

try:
    df = cargar_datos()
    
    # Mostrar un vistazo de los datos para confirmar que cargó bien
    st.success("¡Datos cargados correctamente desde GitHub!")
    
    # Tomamos automáticamente las columnas numéricas para entrenar los modelos
    # (esto evita errores si tus columnas se llaman distinto en el CSV)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Seleccionamos las columnas de forma automática o genérica
    # Asumimos que la última columna o una específica es la calificación a predecir
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columnas_numericas) >= 2:
        X = df[columnas_numericas[:-1]] # Todas menos la última
        Y = df[columnas_numericas[-1]]  # La última columna como objetivo

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Entrenar Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)

        # Entrenar Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        st.subheader("Evaluación de los Modelos")
        st.write(f"**Ridge Regression - R2 Score:** {r2_score(y_test, ridge_pred):.2f}")
        st.write(f"**Random Forest - R2 Score:** {r2_score(y_test, rf_pred):.2f}")

        st.markdown("---")
        st.subheader("Realiza una Predicción")
        
        # Crear controles deslizantes dinámicos basados en las variables del usuario
        valores_usuario = []
        for col in X.columns:
            val_min = float(X[col].min())
            val_max = float(X[col].max())
            val_mean = float(X[col].mean())
            val = st.slider(f"Valor para: {col}", val_min, val_max, val_mean)
            valores_usuario.append(val)

        if st.button("Predecir"):
            entrada = np.array([valores_usuario])
            res_ridge = ridge.predict(entrada)[0]
            res_rf = rf.predict(entrada)[0]

            st.success(f"📌 **Predicción (Ridge Regression):** {res_ridge:.2f}")
            st.success(f"🌲 **Predicción (Random Forest):** {res_rf:.2f}")
    else:
        st.error("El archivo CSV necesita al menos dos columnas numéricas.")

except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
