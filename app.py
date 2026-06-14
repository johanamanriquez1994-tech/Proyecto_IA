import os
import time 
import json 
from datetime import datetime 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 

def preparar_sistema_conocimiento(ruta_manual):
    if not os.path.exists(ruta_manual):
        raise FileNotFoundError(f"No se encontró el manual: {ruta_manual}")
        
    loader = TextLoader(ruta_manual, encoding='utf-8')
    documentos = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    fragmentos = text_splitter.split_documents(documentos)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=fragmentos, embedding=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 2})

def mapear_trazabilidad_pagina(docs):
    contexto_lista = []
    fuentes = set()
    for doc in docs:
        texto = doc.page_content
        contexto_lista.append(texto)
        if "[Página 1]" in texto or "mantenimiento" in texto:
            fuentes.add("- Documento: manual_maquinaria.txt (Página 1)")
        elif "[Página 2]" in texto or "torque" in texto:
            fuentes.add("- Documento: manual_maquinaria.txt (Página 2)")
        elif "[Página 3]" in texto or "mangueras" in texto:
            fuentes.add("- Documento: manual_maquinaria.txt (Página 3)")
    return "\n\n".join(contexto_lista), list(fuentes)

def planificar_prioridad_tarea(pregunta):
    pregunta_min = pregunta.lower()
    if "seguridad" in pregunta_min or "nunca" in pregunta_min or "manipule" in pregunta_min:
        return "CRÍTICA: PROTOCOLO DE SEGURIDAD INDUSTRIAL"
    elif "torque" in pregunta_min or "apriete" in pregunta_min:
        return "ALTA: ESPECIFICACIÓN DE PRECISIÓN MECÁNICA"
    return "NORMAL: RUTINA DE MANTENIMIENTO PREVENTIVO"

def guardar_registro_log(pregunta, prioridad, latencia):
    nombre_archivo = "registro_ejecucion.json"
    
    nuevo_log = {
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pregunta": pregunta,
        "prioridad": prioridad,
        "latencia_segundos": round(latencia, 4)
    }

    if os.path.exists(nombre_archivo):
        with open(nombre_archivo, "r", encoding="utf-8") as f:
            try:
                lista_logs = json.load(f)
            except json.JSONDecodeError:
                lista_logs = []
    else:
        lista_logs = []

    lista_logs.append(nuevo_log)
    
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        json.dump(lista_logs, f, ensure_ascii=False, indent=4)

class AgenteSoporteTerreno:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = Ollama(model="llama3")
        self.historial_global = {} 
        
        self.prompt_sistema = ChatPromptTemplate.from_messages([
            ("system", 
             "Eres un Agente de IA experto en soporte técnico de Industrial-Tech Solutions S.A.\n"
             "Tu misión es asistir mecánicos en terreno usando EXCLUSIVAMENTE el contexto proveído.\n\n"
             "REGLAS ESTRICTAS DE OPERACIÓN:\n"
             "1. Si la información no está en el contexto, di textualmente: 'No cuento con esa información en los manuales internos.' Prohibido inventar.\n"
             "2. Prioriza siempre la seguridad industrial y da respuestas técnicas y directas.\n\n"
             "CONTEXTO TÉCNICO DE HOY:\n{context}\n"
             "PRIORIDAD DE LA SITUACIÓN ACTUAL: {prioridad}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        self.cadena_base = self.prompt_sistema | self.llm | StrOutputParser()

    def obtener_historial_sesion(self, session_id: str):
        if session_id not in self.historial_global:
            self.historial_global[session_id] = ChatMessageHistory()
        return self.historial_global[session_id]

    def consultar(self, session_id: str, pregunta: str):
        tiempo_inicio = time.time()
        
        docs = self.retriever.invoke(pregunta)
        contexto, fuentes = mapear_trazabilidad_pagina(docs)
        
        prioridad = planificar_prioridad_tarea(pregunta)
        
        cadena_con_memoria = RunnableWithMessageHistory(
            self.cadena_base,
            self.obtener_historial_sesion,
            input_messages_key="question",
            history_messages_key="history"
        )
        
        respuesta = cadena_con_memoria.invoke(
            {"question": pregunta, "context": contexto, "prioridad": prioridad},
            config={"configurable": {"session_id": session_id}}
        )
        
        tiempo_final = time.time()
        latencia = tiempo_final - tiempo_inicio
        
        guardar_registro_log(pregunta, prioridad, latencia)
        
        return respuesta, fuentes, prioridad, latencia

if __name__ == "__main__":
    archivo_manual = "manual_maquinaria.txt"
    ID_SESION = "mecanico_faena_01"
    
    try:
        retriever = preparar_sistema_conocimiento(archivo_manual)
        agente = AgenteSoporteTerreno(retriever)
        
        print("\n" + "="*60)
        print("AGENTE DE IA EN LÍNEA - INDUSTRIAL-TECH SOLUTIONS S.A.")
        print("="*60)
        
        pregunta = "¿Qué medida de seguridad se debe tomar con las mangueras de alta presión?"
        
        print(f"\n CONSULTA RECIBIDA: {pregunta}")
        respuesta, fuentes, prioridad, latencia = agente.consultar(ID_SESION, pregunta)
        
        print(f"PLANIFICADOR DE TAREAS: {prioridad}")
        print(f" RESPUESTA COMPORTAMENTAL DEL AGENTE:\n{respuesta}")
        print("TRAZABILIDAD REQUERIDA:")
        for f in fuentes:
            print(f)
            
        print("\n MÉTRICAS DE OBSERVABILIDAD DE LA CONSULTA:")
        print(f"- Latencia de Inferencia: {latencia:.4f} segundos")
        print("="*60)
            
    except Exception as e:
        print(f" Error crítico del sistema: {e}")