import json
import os
import time
import urllib.error
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

if not API_KEY:
    raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY o GOOGLE_API_KEY")


class GeminiRestEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model = model

    def _embed_single(self, text: str) -> list[float]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )
        payload = json.dumps({"content": {"parts": [{"text": text}]}}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["embedding"]["values"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)


def generate_gemini_answer(
    prompt: str,
    model: str | None = None,
    max_retries: int = 3,
) -> str:
    active_model = model or CHAT_MODEL
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if not candidates:
                return "No se obtuvo respuesta del modelo."
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        except urllib.error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise HTTPException(
                status_code=502,
                detail=f"Error al consultar Gemini ({exc.code}): {err_text}",
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise HTTPException(
                status_code=504,
                detail=f"Tiempo de espera agotado al consultar Gemini: {exc}",
            ) from exc

    raise HTTPException(
        status_code=502,
        detail="No se pudo obtener respuesta de Gemini tras varios reintentos.",
    )


app = FastAPI(title="RIC RAG MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

embeddings = GeminiRestEmbeddings(api_key=API_KEY)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


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
            f"[{d.metadata.get('archivo', 'RIC')} - Pág. {d.metadata.get('page', 0) + 1}]\n{d.page_content[:1200]}"
            for d in docs
        ]
    )

    prompt = f"""Eres un asistente técnico de los Pliegos Técnicos Normativos RIC de la SEC (Chile).
Responde ÚNICAMENTE con la información del contexto entregado, de forma clara y concisa.
Cita siempre el archivo/RIC y la página. Si el dato no está en el contexto o depende de una tabla/anexo incompleto, dilo claramente sin inventar.

Contexto:
{context}

Pregunta: {question}"""

    answer = generate_gemini_answer(prompt)

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