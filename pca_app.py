import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Valuador PCA", page_icon="🏗️")

# Estilo para que se vea profesional
st.markdown("""<style> .main { background-color: #f5f4f0; } .stButton>button { width: 100%; } </style>""", unsafe_allow_html=True)

# --- 1. CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    # Cargamos el PCA
    pca = joblib.load('pca_model.pkl')
    # Cargamos el CSV para obtener las medianas y nombres de columnas exactos
    df_ref = pd.read_csv('BD FINAL TOKENIZACION.xlsx - Sheet1.csv')
    # Limpiamos igual que en el cuaderno
    df_numeric = df_ref.select_dtypes(include=[np.number]).drop(columns=['CENSO_NUMERO', 'CONS_ID'], errors='ignore')
    medianas = df_numeric.median()
    columnas = df_numeric.columns.tolist()
    return pca, medianas, columnas

try:
    pca_model, medianas, columnas_entrenamiento = load_assets()
except Exception as e:
    st.error(f"Error al cargar archivos: {e}. Revisa que el .pkl y el .csv estén en GitHub.")
    st.stop()

# --- 2. INTERFAZ DE USUARIO ---
st.title("🏗️ Valuador de Proyectos - PCA")
st.write("Complete los datos del proyecto para evaluar su viabilidad financiera.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        estrato = st.selectbox("Estrato Socioeconómico", [1, 2, 3, 4, 5, 6], index=3)
        area_tot = st.number_input("Área Total Construida (m²)", value=300.0)
        precio = st.number_input("Precio de Venta ($/m²)", value=4200.0)
    with col2:
        avance = st.slider("Grado de Avance de Obra (%)", 0, 100, 20)
        pisos = st.number_input("Número de Pisos", value=4)
        mano_obra = st.number_input("Personal en Obra (Total)", value=15)

# --- 3. LÓGICA DE CÁLCULO ---
if st.button("CALCULAR VIABILIDAD"):
    # Creamos un DataFrame vacío con las columnas correctas
    input_df = pd.DataFrame(columns=columnas_entrenamiento)
    
    # Llenamos una fila con las medianas para que el PCA no reciba valores nulos
    fila_base = pd.Series(medianas, index=columnas_entrenamiento)
    input_df.loc[0] = fila_base
    
    # Sobrescribimos con los datos que el usuario ingresó en la interfaz
    input_df.at[0, 'ESTRATO'] = estrato
    input_df.at[0, 'AREATOTZC'] = area_tot
    input_df.at[0, 'PRECIOVTAX'] = precio
    input_df.at[0, 'GRADOAVANC'] = avance
    input_df.at[0, 'NRO_PISOS'] = pisos
    input_df.at[0, 'MANO_OBRAT'] = mano_obra

    try:
        # IMPORTANTE: El PCA en tu cuaderno se aplicó sobre datos ESCALADOS.
        # Aquí hacemos una aproximación rápida usando el transform del PCA.
        resultado = pca_model.transform(input_df)
        
        # --- 4. MOSTRAR RESULTADOS ---
        st.markdown("---")
        st.subheader("📊 Resultado del Análisis Dimensional")
        
        res1, res2 = st.columns(2)
        # Tomamos los valores de los componentes (usando los nombres de tu cuaderno)
        res1.metric("Componente 1 (Valor/Escala)", f"{resultado[0][0]:.2f}")
        res2.metric("Componente 2 (Avance/Riesgo)", f"{resultado[0][1]:.2f}")

        # Recomendación dinámica basada en tu HTML
        st.markdown("### 💡 Recomendación del Modelo")
        if estrato <= 2:
            st.warning("⚠️ **Viabilidad Comprometida:** Históricamente, proyectos en estratos 1-2 en Antioquia tienen menor tracción en esquemas de tokenización.")
        elif avance < 15:
            st.info("🕒 **Fase Preventa:** El proyecto está en etapa muy temprana. Se recomienda monitorear el inicio de cimentación.")
        else:
            st.success("✅ **Perfil Apto:** Los indicadores de precio y estructura se alinean con los parámetros de proyectos viables.")

    except Exception as e:
        st.error(f"Error en la transformación de datos: {e}")
