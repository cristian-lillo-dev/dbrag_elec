import glob
import os
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric_local")
PDF_DIR = os.getenv("PDF_DIR", "./pdfs_ric")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")


class LocalFastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # E5 rinde mejor con prefijo "passage: " al indexar documentos
        prefixed = [f"passage: {t}" for t in texts]
        return [vec.tolist() for vec in self.model.embed(prefixed)]

    def embed_query(self, text: str) -> list[float]:
        # E5 rinde mejor con prefijo "query: " al consultar
        prefixed = f"query: {text}"
        first = next(self.model.embed([prefixed]))
        return first.tolist()


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

    embeddings = LocalFastEmbedEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Listo: {len(chunks)} fragmentos guardados en {PERSIST_DIR}")


if __name__ == "__main__":
    run_ingest()