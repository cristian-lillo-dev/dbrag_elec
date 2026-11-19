import glob
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# En Railway montaremos el Volume en /data; en local puedes usar ./data
PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
PDF_DIR = os.getenv("PDF_DIR", "./pdfs_ric")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def run_ingest():
    if not API_KEY:
        raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY o GOOGLE_API_KEY")

    docs = []
    for pdf_path in sorted(glob.glob(f"{PDF_DIR}/*.pdf")):
        loader = PyMuPDFLoader(pdf_path)
        for page in loader.load():
            page.metadata["archivo"] = os.path.basename(pdf_path)
            docs.append(page)

    if not docs:
        raise RuntimeError(f"No se encontraron PDFs en {PDF_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=API_KEY,
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Listo: {len(chunks)} fragmentos guardados en {PERSIST_DIR}")


if __name__ == "__main__":
    run_ingest()