import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="Valuador de Tokenización", layout="centered")

@st.cache_resource
def preparar_modelo_y_escalador():
    # 1. Nombre del archivo que DEBE estar en GitHub
    NOMBRE_EXCEL = "BD FINAL TOKENIZACION.xlsx" 
    
    try:
        # 2. Leer el Excel directamente
        df = pd.read_excel(NOMBRE_EXCEL)
        
        # 3. Limpieza de datos (igual a tu cuaderno de Colab)
        # Filtramos solo columnas numéricas y quitamos IDs
        df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['CENSO_NUMERO', 'CONS_ID'], errors='ignore')
        df_numeric = df_numeric.fillna(df_numeric.median())
        
        # 4. Entrenar Escalador (para que el PCA funcione con los nuevos datos)
        scaler = StandardScaler()
        scaler.fit(df_numeric)
        
        # 5. Cargar tu modelo PCA entrenado
        pca = joblib.load('pca_model.pkl')
        
        return pca, scaler, df_numeric.columns.tolist(), df_numeric.median()
    
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo '{NOMBRE_EXCEL}' en el repositorio de GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al procesar los datos: {e}")
        st.stop()

# Cargar activos
pca_model, scaler_model, columnas_modelo, valores_mediana = preparar_modelo_y_escalador()

# --- INTERFAZ GRÁFICA ---
st.title("🏗️ Calculadora de Viabilidad Inmobiliaria")
st.markdown("Basado en el análisis de componentes principales (PCA) para proyectos en Antioquia.")

with st.form("formulario_analisis"):
    st.subheader("Parámetros del Proyecto")
    col1, col2 = st.columns(2)
    
    with col1:
        estrato = st.selectbox("Estrato Socioeconómico", [1, 2, 3, 4, 5, 6], index=3)
        area = st.number_input("Área Total Construida (m²)", value=200.0)
        precio = st.number_input("Precio de Venta ($/m²)", value=4500.0)
    
    with col2:
        avance = st.slider("Porcentaje de Avance (%)", 0, 100, 15)
        pisos = st.number_input("Número de Pisos", value=5, min_value=1)
        mano_obra = st.number_input("Personal en Obra (Total)", value=12)

    enviar = st.form_submit_button("CALCULAR VIABILIDAD")

if enviar:
    # 1. Crear fila con valores promedio (medianas) para las 58 columnas
    input_df = pd.DataFrame([valores_mediana.values], columns=columnas_modelo)
    
    # 2. Actualizar solo las columnas que el usuario ingresó
    input_df['ESTRATO'] = estrato
    input_df['AREATOTZC'] = area
    input_df['PRECIOVTAX'] = precio
    input_df['GRADOAVANC'] = avance
    input_df['NRO_PISOS'] = pisos
    input_df['MANO_OBRAT'] = mano_obra

    # 3. Escalar y transformar con PCA
    datos_escalados = scaler_model.transform(input_df)
    resultado_pca = pca_model.transform(datos_escalados)

    # --- RESULTADOS ---
    st.markdown("---")
    st.success("✅ Análisis finalizado")
    
    res1, res2 = st.columns(2)
    res1.metric("Componente 1 (Valor/Escala)", f"{resultado_pca[0][0]:.2f}")
    res2.metric("Componente 2 (Avance/Riesgo)", f"{resultado_pca[0][1]:.2f}")

    # Recomendación lógica
    st.info("### 📋 Diagnóstico del Modelo")
    if estrato <= 2:
        st.warning("El estrato reportado presenta desafíos históricos de rentabilidad para tokenización.")
    elif avance < 20:
        st.write("El proyecto se encuentra en una fase de riesgo constructivo moderado debido a su etapa temprana.")
    else:
        st.write("El perfil del proyecto muestra una correlación positiva con los modelos de éxito financiero en la región.")
