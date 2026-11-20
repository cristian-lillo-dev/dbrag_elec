import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
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


def _recortar_limpio(texto: str, limite: int = 950) -> str:
    texto = texto.strip()
    if len(texto) <= limite:
        return texto
    recorte = texto[:limite]
    ultimo_espacio = recorte.rfind(" ")
    if ultimo_espacio > int(limite * 0.7):
        recorte = recorte[:ultimo_espacio]
    return recorte + "..."


def _limpiar_html_respuesta(raw_text: str) -> str:
    texto = raw_text.strip()
    # Quita posibles cercas ```html ... ``` si el modelo las incluyó
    texto = re.sub(r"^```(?:html)?\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s*```$", "", texto)

    # Elimina etiquetas peligrosas o de documento completo por seguridad
    texto = re.sub(
        r"</?(?:script|style|iframe|object|embed|link|meta|html|head|body)[^>]*>",
        "",
        texto,
        flags=re.IGNORECASE,
    )
    texto = re.sub(
        r"\s+on\w+\s*=\s*(['\"]).*?\1",
        "",
        texto,
        flags=re.IGNORECASE,
    )

    # Si no contiene etiquetas HTML básicas, convierte negritas/párrafos simples
    if not re.search(r"<(?:h3|h4|p|ul|ol|li|strong|div)\b", texto, re.IGNORECASE):
        escapado = html.escape(texto)
        escapado = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escapado)
        parrafos = [
            f"<p>{p.replace(chr(10), '<br>')}</p>"
            for p in escapado.split("\n\n")
            if p.strip()
        ]
        return "".join(parrafos) if parrafos else "<p>Sin respuesta del modelo.</p>"

    return texto


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
                    "Responde ÚNICAMENTE con la información del contexto entregado, de forma clara, técnica y estructurada. "
                    "Devuelve EXCLUSIVAMENTE un fragmento HTML válido usando solo las etiquetas: "
                    "<h3>, <p>, <ul>, <ol>, <li>, <strong> y <br>. "
                    "NO uses Markdown (nada de **, #, -, ni ```html). "
                    "NO incluyas <html>, <head> ni <body>. "
                    "NO repitas la fuente ni el nombre del PDF dentro de cada viñeta, porque las fuentes se muestran aparte. "
                    "Organiza la respuesta con un <h3> breve del tema y listas <ul>/<ol> con <strong> en los conceptos clave. "
                    "Si el dato no está en el contexto o depende de una tabla/anexo incompleto, indícalo en un <p> sin inventar."
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
                return "<p>No se obtuvo respuesta del modelo.</p>"
            message = choices[0].get("message", {})
            contenido = (message.get("content") or "").strip()
            if not contenido:
                return "<p>No se obtuvo respuesta del modelo.</p>"
            return _limpiar_html_respuesta(contenido)
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

embeddings = LocalFastEmbedEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/ric/docs")
def list_ingested_docs():
    registro_path = os.path.join(PERSIST_DIR, "registro_ingesta.json")
    if not os.path.exists(registro_path):
        return {
            "mensaje": (
                "Aún no existe registro_ingesta.json. Ejecuta ingest.py"
                " primero."
            ),
            "archivos": {},
        }
    with open(registro_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/ric/ask")
def ask_ric(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="La pregunta no puede estar vacía."
        )

    docs = retriever.invoke(question)
    if not docs:
        return {
            "answer": (
                "<p>No encontré información relevante en la base de datos RIC"
                " para esa consulta.</p>"
            ),
            "sources": [],
        }

    context = "\n\n---\n\n".join([
        (
            f"[{d.metadata.get('archivo', 'RIC')} - Pág."
            f" {d.metadata.get('page', 0) + 1}]\n{_recortar_limpio(d.page_content, 950)}"
        )
        for d in docs
    ])

    prompt = f"""Contexto normativo:
{context}

Pregunta del usuario: {question}"""

    answer_html = generate_groq_answer(prompt)

    sources = []
    seen = set()
    for d in docs:
        archivo = d.metadata.get("archivo", "RIC")
        pagina = d.metadata.get("page", 0) + 1
        key = (archivo, pagina)
        if key not in seen:
            seen.add(key)
            sources.append({"archivo": archivo, "pagina": pagina})

    return {"answer": answer_html, "sources": sources}