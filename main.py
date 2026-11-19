import json
import os
import time
import urllib.error
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric_local")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

if not GROQ_API_KEY:
    raise RuntimeError("Falta la variable de entorno GROQ_API_KEY")


class LocalFastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def _prepare(self, texts: list[str], kind: str) -> list[str]:
        if "e5" in self.model_name.lower():
            prefix = "passage: " if kind == "doc" else "query: "
            return [f"{prefix}{t}" for t in texts]
        return texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = self._prepare(texts, "doc")
        return [vec.tolist() for vec in self.model.embed(prepared)]

    def embed_query(self, text: str) -> list[float]:
        prepared = self._prepare([text], "query")
        first = next(self.model.embed(prepared))
        return first.tolist()


def generate_groq_answer(
    prompt: str,
    model: str | None = None,
    max_retries: int = 3,
) -> str:
    active_model = model or GROQ_CHAT_MODEL
    payload = {
        "model": active_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente técnico de los Pliegos Técnicos Normativos RIC de la SEC (Chile). "
                    "Responde ÚNICAMENTE con la información del contexto entregado, de forma clara y concisa. "
                    "Cita siempre el archivo/RIC y la página. Si el dato no está en el contexto o depende de "
                    "una tabla/anexo incompleto, dilo claramente sin inventar."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    raw_payload = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(
            GROQ_CHAT_URL,
            data=raw_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "User-Agent": "RIC-RAG-MVP/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            choices = data.get("choices", [])
            if not choices:
                return "No se obtuvo respuesta del modelo."
            message = choices[0].get("message", {})
            return (message.get("content") or "").strip() or "No se obtuvo respuesta del modelo."
        except urllib.error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise HTTPException(
                status_code=502,
                detail=f"Error al consultar Groq ({exc.code}): {err_text}",
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise HTTPException(
                status_code=504,
                detail=f"Tiempo de espera agotado al consultar Groq: {exc}",
            ) from exc

    raise HTTPException(
        status_code=502,
        detail="No se pudo obtener respuesta de Groq tras varios reintentos.",
    )


app = FastAPI(title="RIC RAG MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

embeddings = LocalFastEmbedEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


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
            f"[{d.metadata.get('archivo', 'RIC')} - Pág. {d.metadata.get('page', 0) + 1}]\n{d.page_content[:850]}"
            for d in docs
        ]
    )

    prompt = f"""Contexto:
{context}

Pregunta: {question}"""

    answer = generate_groq_answer(prompt)

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