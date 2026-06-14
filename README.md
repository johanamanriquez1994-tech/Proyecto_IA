#  Sistema de Observabilidad y Soporte de Agentes con IA
**Duoc UC - Optativo Ingeniería de Soluciones con IA (ISY0101)**
**Evaluación Parcial N°3**

Este repositorio contiene la evolución del Agente de IA para soporte de mecánicos en terreno (*Industrial-Tech Solutions S.A.*), integrando herramientas de observabilidad, auditoría de logs y un panel visual interactivo.

---

##  Historial de Modificaciones (Entrega Actual)

En esta tercera fase del proyecto, se realizaron las siguientes modificaciones estructurales sobre la versión previa:

1. **Refactorización en `app.py` (Métricas de Latencia):** Se incorporó el módulo nativa `time` para implementar un cronómetro de precisión (`time.time()`). Ahora la función `consultar` calcula y retorna los segundos exactos de inferencia de Llama 3.
2. **Sistema de Auditoría Persistente:** Se diseñó la función `guardar_registro_log` utilizando la librería `json`. El agente escribe automáticamente cada interacción de forma histórica sin sobrescribir los datos previos.
3. **Creación de `dashboard.py` (Nueva Incorporación):** Archivo totalmente nuevo que utiliza **Streamlit** y **Pandas** para leer de manera asíncrona el archivo de logs y proyectar gráficos estadísticos interactivos de rendimiento.

---

##  Requisitos e Instalación

Para ejecutar este proyecto, asegúrese de instalar las siguientes dependencias:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-chroma streamlit pandas huggingface-hub