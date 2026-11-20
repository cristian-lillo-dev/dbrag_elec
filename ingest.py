from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from chromadb.config import Settings
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
PDF_DIR = os.getenv("PDF_DIR", "./pdfs_ric")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


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


def calcular_sha256(ruta_archivo: str) -> str:
    h = hashlib.sha256()
    with open(ruta_archivo, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def guardar_registro(
    pdf_files: list[str],
    docs: list,
    chunks: list,
) -> str:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    registro_path = os.path.join(PERSIST_DIR, "registro_ingesta.json")

    registro_anterior = {}
    if os.path.exists(registro_path):
        with open(registro_path, "r", encoding="utf-8") as f:
            registro_anterior = json.load(f)

    archivos_anteriores = registro_anterior.get("archivos", {})
    archivos_info = {}

    for path in pdf_files:
        nombre = os.path.basename(path)
        sha = calcular_sha256(path)
        n_paginas = sum(1 for d in docs if d.metadata.get("archivo") == nombre)
        n_chunks = sum(1 for c in chunks if c.metadata.get("archivo") == nombre)
        prev = archivos_anteriores.get(nombre)

        if not prev:
            estado = "nuevo"
        elif prev.get("sha256") != sha:
            estado = "modificado"
        else:
            estado = "sin_cambios"

        archivos_info[nombre] = {
            "sha256": sha,
            "paginas": n_paginas,
            "fragmentos": n_chunks,
            "estado_ultima_ingesta": estado,
        }

    eliminados = [
        nombre for nombre in archivos_anteriores if nombre not in archivos_info
    ]

    data_registro = {
        "ultima_actualizacion_utc": datetime.now(timezone.utc).isoformat(),
        "persist_dir": PERSIST_DIR,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_archivos": len(pdf_files),
        "total_fragmentos": len(chunks),
        "eliminados_desde_anterior": eliminados,
        "archivos": archivos_info,
    }

    with open(registro_path, "w", encoding="utf-8") as f:
        json.dump(data_registro, f, indent=2, ensure_ascii=False)

    return registro_path


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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total fragmentos a vectorizar: {len(chunks)}")

    embeddings = LocalFastEmbedEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        client_settings=Settings(anonymized_telemetry=False),
    )

    registro_path = guardar_registro(pdf_files, docs, chunks)
    print(f"Listo: {len(chunks)} fragmentos guardados en {PERSIST_DIR}")
    print(f"Registro actualizado en: {registro_path}")


if __name__ == "__main__":
    run_ingest()