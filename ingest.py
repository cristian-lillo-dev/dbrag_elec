from datetime import datetime, timezone
import glob
import hashlib
import json
import os
import re
from chromadb.config import Settings
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
PDF_DIR = os.getenv("PDF_DIR", "./pdfs_ric")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "850"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

RIC_PATTERN = re.compile(
    r"\bric\s*[-_–—]?\s*n?\s*[°º]?\s*[-_–—]?\s*0*(\d+)\b",
    re.IGNORECASE,
)


def extraer_alias_ric(texto: str) -> list[str]:
    """Devuelve variantes canónicas de cualquier mención RIC encontrada en el texto."""
    numeros = sorted({match.group(1) for match in RIC_PATTERN.finditer(texto)})
    aliases: list[str] = []
    for num in numeros:
        aliases.extend([
            f"RIC-N{num}",
            f"RIC N° {num}",
            f"RIC n° {num}",
            f"RIC {num}",
            f"RIC-{num}",
        ])
    # Preserva orden sin duplicados
    return list(dict.fromkeys(aliases))


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


def preparar_chunks(docs: list[Document]) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    chunks_enriquecidos: list[Document] = []
    ids: list[str] = []
    contadores_por_archivo: dict[str, int] = {}

    for d in docs:
        archivo = d.metadata.get("archivo", "documento.pdf")
        pagina = int(d.metadata.get("page", 0))
        sub_chunks = splitter.split_documents([d])

        for sc in sub_chunks:
            contenido_limpio = re.sub(r"[ \t]+", " ", sc.page_content).strip()
            if not contenido_limpio:
                continue

            idx = contadores_por_archivo.get(archivo, 0)
            contadores_por_archivo[archivo] = idx + 1

            aliases = extraer_alias_ric(f"{archivo} {contenido_limpio[:250]}")
            etiqueta_ric = f" | Alias: {', '.join(aliases)}" if aliases else ""
            encabezado = (
                f"[Archivo: {archivo} | Pág. {pagina + 1}{etiqueta_ric}]\n"
            )

            metadata = dict(sc.metadata)
            metadata["archivo"] = archivo
            metadata["page"] = pagina
            metadata["chunk_index"] = idx
            if aliases:
                metadata["ric_aliases"] = ", ".join(aliases)

            chunks_enriquecidos.append(
                Document(
                    page_content=f"{encabezado}{contenido_limpio}",
                    metadata=metadata,
                )
            )
            ids.append(f"{archivo}::p{pagina}::c{idx}")

    return chunks_enriquecidos, ids


def guardar_registro(
    pdf_files: list[str],
    docs: list[Document],
    chunks: list[Document],
    estados_archivos: dict[str, str],
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

        archivos_info[nombre] = {
            "sha256": sha,
            "paginas": n_paginas,
            "fragmentos": n_chunks,
            "estado_ultima_ingesta": estados_archivos.get(nombre, "en_chroma"),
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

    registro_path = os.path.join(PERSIST_DIR, "registro_ingesta.json")
    registro_anterior = {}
    if os.path.exists(registro_path):
        with open(registro_path, "r", encoding="utf-8") as f:
            registro_anterior = json.load(f)
    archivos_anteriores = registro_anterior.get("archivos", {})

    embeddings = LocalFastEmbedEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )

    archivos_actuales = {os.path.basename(p) for p in pdf_files}
    for eliminado in set(archivos_anteriores.keys()) - archivos_actuales:
        vectorstore.delete(where={"archivo": eliminado})

    todos_docs: list[Document] = []
    chunks_a_upsert: list[Document] = []
    ids_a_upsert: list[str] = []
    estados_archivos: dict[str, str] = {}

    for path in pdf_files:
        nombre = os.path.basename(path)
        sha = calcular_sha256(path)
        prev = archivos_anteriores.get(nombre)

        if not prev:
            estado = "nuevo"
        elif prev.get("sha256") != sha:
            estado = "modificado"
        else:
            estado = "sin_cambios"

        estados_archivos[nombre] = estado

        loader = PyMuPDFLoader(path)
        loaded = loader.load()
        for d in loaded:
            d.metadata["archivo"] = nombre
        todos_docs.extend(loaded)

        # Re-indexa archivos nuevos, modificados o que aún no tenían fragmentos
        if estado in ("nuevo", "modificado") or prev.get("fragmentos", 0) == 0:
            vectorstore.delete(where={"archivo": nombre})
            file_chunks, file_ids = preparar_chunks(loaded)
            chunks_a_upsert.extend(file_chunks)
            ids_a_upsert.extend(file_ids)

    if chunks_a_upsert:
        print(f"Actualizando {len(chunks_a_upsert)} fragmentos en Chroma...")
        vectorstore.add_documents(documents=chunks_a_upsert, ids=ids_a_upsert)
    else:
        print("No hay archivos nuevos o modificados para re-vectorizar.")

    # Cuenta total de fragmentos vigentes a partir de los archivos procesados
    chunks_totales_estimados: list[Document] = []
    for path in pdf_files:
        nombre = os.path.basename(path)
        docs_archivo = [d for d in todos_docs if d.metadata.get("archivo") == nombre]
        c_archivo, _ = preparar_chunks(docs_archivo)
        chunks_totales_estimados.extend(c_archivo)

    reg_path = guardar_registro(
        pdf_files, todos_docs, chunks_totales_estimados, estados_archivos
    )
    print(f"Listo. Registro actualizado en: {reg_path}")


if __name__ == "__main__":
    run_ingest()