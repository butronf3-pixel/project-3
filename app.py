import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página en modo ancho para mejor visualización
st.set_page_config(page_title="Predictor Académico UAP", page_icon="📊", layout="wide")

st.title("📊 Predictor de Rendimiento Académico y Comportamiento")
st.markdown("### Proyecto 3 - Métodos Estadísticos (Ridge Regression & Random Forest)")
st.markdown("---")

@st.cache_data
def cargar_datos():
    df = pd.read_csv('datos_estudiantes.csv', encoding='latin1')
    return df

try:
    df = cargar_datos()
    
    # Limpieza automática de caracteres extraños y errores de codificación
    df.columns = df.columns.str.replace('ÿ', 'i', regex=False).str.replace('', 'i', regex=False)
    
    # Seleccionar solo columnas numéricas y descartar el ID si existe
    df_num = df.select_dtypes(include=[np.number]).copy()
    if 'ID' in df_num.columns:
        df_num = df_num.drop(columns=['ID'])
    
    df_num = df_num.dropna(how='any')

    if df_num.shape[1] >= 2:
        # Definir X (variables independientes) y Y (variable objetivo)
        X = df_num.iloc[:, :-1]
        Y = df_num.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Entrenar Modelos
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # Diseño de la interfaz en dos columnas principales
        col_izq, col_der = st.columns([1.3, 1], gap="large")

        with col_izq:
            st.subheader("🎛️ Simulador de Variables y Hábitos")
            st.write("Ajusta los valores para simular el comportamiento del estudiante:")
            
            valores_usuario = []
            for col in X.columns:
                # Limpiar nombre de la columna para que se lea impecable
                nombre_limpio = col.replace("?", "").replace("What was your grade the last period? (Specify) Example: 6.7", "Calificación periodo anterior").strip()
                val_min = float(X[col].min())
                val_max = float(X[col].max())
                val_mean = float(X[col].mean())
                
                val = st.slider(f"**{nombre_limpio}**", val_min, val_max, val_mean)
                valores_usuario.append(val)

        with col_der:
            st.subheader("📈 Rendimiento de los Modelos")
            
            # Métricas de precisión (R² Score)
            m1, m2 = st.columns(2)
            m1.metric("Ridge R²", f"{r2_score(y_test, ridge_pred):.2f}")
            m2.metric("Random Forest R²", f"{r2_score(y_test, rf_pred):.2f}")
            
            st.markdown("---")
            st.subheader("🎯 Resultado de la Predicción")
            
            if st.button("🚀 Calcular Predicción", use_container_width=True):
                entrada = np.array([valores_usuario])
                res_ridge = ridge.predict(entrada)[0]
                res_rf = rf.predict(entrada)[0]

                res_c1, res_c2 = st.columns(2)
                res_c1.metric("Modelo Ridge", f"{res_ridge:.2f}")
                res_c2.metric("Random Forest", f"{res_rf:.2f}")
                
                st.success("¡Predicción calculada correctamente con ambos algoritmos!")

    else:
        st.error("No se encontraron suficientes columnas numéricas válidas para entrenar los modelos.")

except Exception as e:
    st.error(f"Ocurrió un error al procesar los datos: {e}")
