# Agente de IA Local de Soporte Técnico - Industrial-Tech Solutions S.A.

Este repositorio contiene el prototipo de un Agente de Inteligencia Artificial Autónomo con arquitectura RAG (Retrieval-Augmented Generation) diseñado para asistir a mecánicos automotrices en terreno de forma 100% local.

##  Diagrama de Orquestación de Componentes (IE7)

El flujo de información y acoplamiento de tecnologías del agente sigue la siguiente estructura lógica:



[manual_maquinaria.txt] ──> [Text Loader & Splitter] ──> [HuggingFace Embeddings]
│
▼
[Consulta del Mecánico] ──> [Planificador de Tareas] ──> [ChromaDB Vector Store]
│                                                       │
▼                                                       ▼
[Historial de Conversación] ───────────────────────────> [Motor LLM Llama 3]
│
▼
[Respuesta con Trazabilidad]

## Frameworks e Integraciones (IE2, IE3 y IE4)
**Framework Central:** LangChain, utilizado para estructurar las cadenas de ejecución (Chains) de forma modular y compatible.
 **Base de Datos Vectorial:** ChromaDB, encargada de indexar y recuperar el contexto técnico relevante en milisegundos.
 **Modelo de Embeddings:** `all-MiniLM-L6-v2` de HuggingFace, ocupado para convertir el texto técnico en densidades vectoriales matemáticas.
 **Modelo de Lenguaje (LLM):** Ollama ejecutando Llama 3 de manera local, garantizando la privacidad y confidencialidad de los manuales de la empresa.
 **Memoria de Contenido:** Conexión de historial conversacional para dar continuidad y retención a flujos prolongados en terreno.

## Reglas Operativas e Inferencia (IE5 y IE6)
1 **Planificación Estricta:** Clasificación previa de la consulta en tres niveles de criticidad (Normal, Alta, Crítica) para secuenciar prioridades operativas.
2 **Protocolo Anti-Alucinación:** Restricción mediante Prompt del Sistema que fuerza al agente a declarar desconocimiento si el dato solicitado no existe explícitamente en el manual técnico.