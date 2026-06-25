 Agente de Soporte Técnico Autónomo con Arquitectura RAG
**Duoc UC - Optativo Ingeniería de Soluciones con IA (ISY0101)**


Este repositorio contiene la implementación de un Agente de IA local diseñado para asistir a mecánicos en terreno (*Industrial-Tech Solutions S.A.*). El sistema integra recuperación semántica de manuales técnicos, memoria conversacional persistente y herramientas de ejecución autónoma.


  Diagrama de Orquestación de Componentes (IE7)

El flujo de trabajo e interconexión de los módulos del agente sigue la siguiente estructura lógica:

[ Mecánico en Terreno (Usuario) ]
│
   (Ingresa Consulta Semántica)
┌─────────────────────────────────────────────────────────┐
│ 1. Filtro Perimetral & Planificador de Tareas (IE5)     │
│    - Clasifica Prioridad: CRÍTICA, ALTA, NORMAL        │
└──────────────────────────┬──────────────────────────────┘
│
┌───────────────┴───────────────┐
                               
┌───────────────────────────┐   ┌───────────────────────────┐
│ 2. Recuperador RAG (IE4)  │   │ 3. Herramienta (IE1)      │
│   - TextLoader (Manual)   │   │   - Ejecución Autónoma    │
│   - Chroma Vector Store   │   │   - Cálculo de Holgura    │
│   - Embeddings MiniLM     │   │     Mantenimiento Técnico │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
│                               │
└───────────────┬───────────────┘
  (Contexto + Datos de Tool)
┌───────────────────────────────────────────────────────────┐
│ 4. Orquestador Central: LangChain + Memory (IE2, IE3)     │
│    - RunnableWithMessageHistory / ChatMessageHistory       │
└─────────────────────────────┬─────────────────────────────┘
│
  (Prompt Enriquecido)
┌───────────────────────────────────────────────────────────┐
│ 5. Motor de Inferencia Local: Ollama (IE2)                │
│    - Modelo Principal: Llama 3                            │
└─────────────────────────────┬─────────────────────────────┘
│
  (Respuesta Estructurada)
[ Solución Técnico Validada / Toma de Decisiones (IE6) ]

 Requisitos Técnicos e Instalación (IE2)

El agente está construido sobre el ecosistema de **LangChain** y utiliza procesamiento local para resguardar la privacidad de los manuales industriales:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-chroma huggingface-hub