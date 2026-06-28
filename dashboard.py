import streamlit as st
import pandas as pd
import os


st.set_page_config(page_title="Industrial-Tech Solutions - Telemetría IA", layout="wide")

st.title(" Panel de Observabilidad y Rendimiento del Agente RAG")
st.markdown("### Enfoque de Telemetría Local y Privacidad Industrial")
st.write("Este panel visualiza las métricas de rendimiento y trazabilidad operativa recolectadas localmente desde el motor del agente.")

CSV_FILE = "registro_ejecucion.csv"

if not os.path.exists(CSV_FILE):
    st.error(f"No se encontró el archivo de logs '{CSV_FILE}'. Ejecuta primero 'python app.py' para generar datos.")
else:
  
    df = pd.read_csv(CSV_FILE)
    
  
    st.subheader(" Indicadores Clave de Rendimiento (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    total_consultas = len(df)
    latencia_promedio = df["Latencia_Segundos"].mean()
    consistencia_promedio = df["Tokens_Caracteres"].mean()
    tasa_errores = (df["Status_Error"].sum() / total_consultas) * 100
    
    col1.metric(label="Total Consultas Auditadas", value=f"{total_consultas} eventos")
    col2.metric(label="Latencia Promedio (Llama 3)", value=f"{latencia_promedio:.2f} seg")
    col3.metric(label="Consistencia (Tamaño Promedio)", value=f"{int(consistencia_promedio)} carac.")
    col4.metric(label="Tasa de Error Global", value=f"{tasa_errores:.1f} %")
    
    st.markdown("---")

    st.subheader(" Análisis de Distribución y Rendimiento")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Comportamiento de la Latencia por Consulta en Segundos (IE2)**")
        
        st.bar_chart(data=df, y="Latencia_Segundos")
        
    with col_g2:
        st.markdown("**Consistencia de Respuestas en Caracteres (IE1)**")
   
        st.line_chart(data=df, y="Tokens_Caracteres")
        
    st.markdown("---")
    
  
    st.subheader(" Registro Histórico de Auditoría y Eventos")
    st.write("Filtrado completo de eventos en tiempo real para análisis de cuellos de botella:")
    
    st.dataframe(df[["Timestamp", "Session_ID", "Pregunta", "Prioridad", "Latencia_Segundos", "Status_Error"]], 
                 use_container_width=True)