import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

if not API_KEY:
    raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY o GOOGLE_API_KEY")

app = FastAPI(title="RIC RAG MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=API_KEY,
    transport="rest",
)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=API_KEY,
    transport="rest",
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/ric/ask")
def ask_ric(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    docs = retriever.invoke(question)
    if not docs:
        return {
            "answer": "No encontré información relevante en la base de datos RIC para esa consulta.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        [
            f"[{d.metadata.get('archivo', 'RIC')} - Pág. {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
            for d in docs
        ]
    )

    prompt = f"""Eres un asistente técnico de los Pliegos Técnicos Normativos RIC de la SEC (Chile).
Responde ÚNICAMENTE con la información del contexto entregado.
Cita siempre el archivo/RIC y la página. Si el dato no está en el contexto o depende de una tabla/anexo incompleto, dilo claramente sin inventar.

Contexto:
{context}

Pregunta: {question}"""

    answer = llm.invoke(prompt).content

    sources = []
    seen = set()
    for d in docs:
        archivo = d.metadata.get("archivo", "RIC")
        pagina = d.metadata.get("page", 0) + 1
        key = (archivo, pagina)
        if key not in seen:
            seen.add(key)
            sources.append({"archivo": archivo, "pagina": pagina})

    return {"answer": answer, "sources": sources}