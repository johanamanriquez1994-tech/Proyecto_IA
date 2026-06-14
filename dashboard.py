import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(page_title="Dashboard de Observabilidad IA", layout="wide")

st.title(" Panel de Observabilidad y Métricas de Rendimiento")
st.subheader("Industrial-Tech Solutions S.A. - Soporte de Agentes en Terreno")
st.markdown("---")

nombre_log = "registro_ejecucion.json"

if os.path.exists(nombre_log):
    with open(nombre_log, "r", encoding="utf-8") as f:
        try:
            datos = json.load(f)
        except json.JSONDecodeError:
            datos = []
            
    if datos:
        df = pd.DataFrame(datos)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=" Total Consultas Registradas", value=len(df))
        with col2:
            st.metric(label=" Latencia Promedio", value=f"{df['latencia_segundos'].mean():.2f} seg")
        with col3:
            st.metric(label=" Latencia Mínima (Óptima)", value=f"{df['latencia_segundos'].min():.2f} seg")
            
        st.markdown("---")
        
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.subheader(" Evolución de la Latencia por Consulta")
            st.line_chart(df['latencia_segundos'])
            
        with col_graf2:
            st.subheader("Distribución por Prioridad de Tarea")
            conteo_prioridades = df['prioridad'].value_counts()
            st.bar_chart(conteo_prioridades)
            
        st.markdown("---")
        
        st.subheader(" Historial de Logs en Tiempo Real (Trazabilidad)")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.warning("El archivo de registros está vacío. Ejecuta tu app.py para generar datos.")
else:
    st.error(f"No se encontró el archivo '{nombre_log}'. Ejecuta app.py primero para crearlo.")