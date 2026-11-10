import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
    
    # Limpiar nombres de columnas de caracteres extraños
    df.columns = df.columns.str.replace('ÿ', 'i', regex=False)

    # Seleccionar únicamente columnas que contengan números de forma segura
    # Rellenamos los espacios vacíos con ceros o con la media para no perder filas
    df_num = df.select_dtypes(include=[np.number]).copy()
    
    # Si la columna ID está presente, la quitamos para que no afecte la predicción
    if 'ID' in df_num.columns:
        df_num = df_num.drop(columns=['ID'])

    # Rellenar valores nulos (NaN) con la mediana de cada columna en lugar de borrarlos
    df_num = df_num.fillna(df_num.median())

    if df_num.shape[0] > 5 and df_num.shape[1] >= 2:
        # Definir X (variables predictoras) y Y (última columna numérica como objetivo)
        X = df_num.iloc[:, :-1]
        Y = df_num.iloc[:, -1]

        # Ajustar el test_size de manera segura según el número de filas disponibles
        test_s = 0.2 if len(df_num) > 10 else 0.3
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_s, random_state=42)

        # Entrenar Modelos
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # Diseño de la interfaz en dos columnas
        col_izq, col_der = st.columns([1.3, 1], gap="large")

        with col_izq:
            st.subheader("🎛️ Simulador de Variables y Hábitos")
            st.write("Ajusta los valores para simular el comportamiento:")
            
            valores_usuario = []
            for col in X.columns:
                nombre_limpio = col.replace("?", "").strip()
                val_min = float(X[col].min())
                val_max = float(X[col].max())
                val_mean = float(X[col].mean())
                
                # Evitar rangos inválidos si el min y max son iguales
                if val_min == val_max:
                    val_max = val_min + 1.0

                val = st.slider(f"**{nombre_limpio}**", val_min, val_max, val_mean)
                valores_usuario.append(val)

        with col_der:
            st.subheader("📈 Rendimiento de los Modelos")
            
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
                
                st.success("¡Predicción calculada con éxito!")

    else:
        st.error("El archivo no tiene suficientes filas o columnas numéricas para procesar los modelos.")

except Exception as e:
    st.error(f"Ocurrió un error al procesar los datos: {e}")
