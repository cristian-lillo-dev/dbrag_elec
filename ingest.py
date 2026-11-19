import glob
import json
import os
import time
import urllib.error
import urllib.request
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# En Railway montaremos el Volume en /data; en local puedes usar ./data
PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
PDF_DIR = os.getenv("PDF_DIR", "./pdfs_ric")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY o GOOGLE_API_KEY")


class GeminiRestEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model = model

    def _embed_single(self, text: str, max_retries: int = 4) -> list[float]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )
        payload = json.dumps({"content": {"parts": [{"text": text}]}}).encode("utf-8")

        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["embedding"]["values"]
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < max_retries - 1:
                    wait_sec = 5 * (attempt + 1)
                    print(f"Rate limit (429), reintentando en {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Error HTTP {exc.code} en embeddings: {body}") from exc

        raise RuntimeError("No se pudo generar embedding tras varios reintentos.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        total = len(texts)
        for i, text in enumerate(texts, start=1):
            vectors.append(self._embed_single(text))
            if i % 20 == 0 or i == total:
                print(f"Procesados {i}/{total} fragmentos...")
                time.sleep(0.5)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)


def run_ingest():
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron PDFs en {PDF_DIR}")

    docs = []
    for path in pdf_files:
        loader = PyMuPDFLoader(path)
        loaded = loader.load()
        for d in loaded:
            d.metadata["archivo"] = os.path.basename(path)
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total fragmentos a vectorizar: {len(chunks)}")

    embeddings = GeminiRestEmbeddings(api_key=API_KEY)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Listo: {len(chunks)} fragmentos guardados en {PERSIST_DIR}")


if __name__ == "__main__":
    run_ingest()