import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.tools import tool

def preparar_sistema_conocimiento(ruta_manual):
    if not os.path.exists(ruta_manual):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_manual = os.path.join(base_dir, ruta_manual)
        if not os.path.exists(ruta_manual):
            raise FileNotFoundError(f"No se encontró el manual en ninguna ruta: {ruta_manual}")
        
    loader = TextLoader(ruta_manual, encoding='utf-8')
    documentos = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    fragmentos = text_splitter.split_documents(documentos)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=fragmentos, embedding=embeddings)

    return vector_store.as_retriever(search_kwargs={"k": 2})


@tool
def calcular_holgura_mantenimiento(horas_actuales: int) -> str:
    """Calcula cuántas horas restan o si se excedió el tiempo límite para el mantenimiento preventivo."""
    LIMITE_HORAS = 250
    if horas_actuales < LIMITE_HORAS:
        restantes = LIMITE_HORAS - horas_actuales
        return f"El equipo dispone de una holgura segura. Faltan {restantes} horas de operación para el mantenimiento preventivo."
    elif horas_actuales == LIMITE_HORAS:
        return "Alerta: El equipo ha alcanzado exactamente las 250 horas. Se debe detener y agendar mantenimiento inmediato."
    else:
        exceso = horas_actuales - LIMITE_HORAS
        return f"CRÍTICO: El equipo presenta un exceso de {exceso} horas sobre el límite permitido de operación sin mantenimiento."


def planificar_prioridad_tarea(pregunta: str) -> str:
    pregunta_min = pregunta.lower()
    if "seguridad" in pregunta_min or "nunca" in pregunta_min or "manipule" in pregunta_min:
        return "CRÍTICA: PROTOCOLO DE SEGURIDAD INDUSTRIAL"
    elif "torque" in pregunta_min or "apriete" in pregunta_min or "pernos" in pregunta_min:
        return "ALTA: ESPECIFICACIÓN DE PRECISIÓN MECÁNICA"
    return "NORMAL: RUTINA DE MANTENIMIENTO PREVENTIVO"

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
        docs = self.retriever.invoke(pregunta)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        
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
        
        return respuesta, prioridad


if __name__ == "__main__":
    archivo_manual = "manual_maquinaria.txt"
    ID_SESION = "mecanico_evaluacion_02"
    
    try:
        retriever = preparar_sistema_conocimiento(archivo_manual)
        agente = AgenteSoporteTerreno(retriever)
        
        print("\n" + "="*60)
        print("DEMOSTRACIÓN DE EVALUACIÓN N°2 - AGENTE DE IA")
        print("="*60)
        
        ejemplo_1 = "¿Qué medida de seguridad se debe tomar con las mangueras de alta presión?"
        print(f"\n[Ejemplo 1] Consulta del Operador: {ejemplo_1}")
        respuesta_1, prioridad_1 = agente.consultar(ID_SESION, ejemplo_1)
        print(f"-> Esquema de Planificación (IE5): {prioridad_1}")
        print(f"-> Comportamiento / Decisión del Agente (IE6):\n{respuesta_1}")
        print("-" * 60)

        print("\n[Ejemplo 2] Ejecución Autónoma de Herramienta Técnica:")
        horas_test = 265
        resultado_herramienta = calcular_holgura_mantenimiento.invoke({"horas_actuales": horas_test})
        print(f"-> Horas registradas en terreno: {horas_test} hrs.")
        print(f"-> Respuesta de la herramienta autónoma (IE1): {resultado_herramienta}")
        print("="*60)
            
    except Exception as e:
        print(f"Error en la ejecución de la arquitectura: {e}")