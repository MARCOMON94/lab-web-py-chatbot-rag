import os
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DOCS_DIR = Path("docs")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documentos_rag")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COSTE_POR_1000_TOKENS = 0.0


openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def contar_tokens(texto: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(texto))


def leer_documentos() -> list[Path]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError("No existe la carpeta docs/")

    archivos = list(DOCS_DIR.glob("*.txt"))

    if len(archivos) == 0:
        raise FileNotFoundError("No hay archivos .txt dentro de docs/")

    return archivos


def dividir_en_chunks(texto: str) -> list[str]:
    palabras = texto.split()
    chunks = []

    inicio = 0

    while inicio < len(palabras):
        fin = inicio + CHUNK_SIZE
        chunk = " ".join(palabras[inicio:fin])

        if chunk.strip():
            chunks.append(chunk)

        inicio += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def crear_embedding(texto: str) -> list[float]:
    respuesta = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto
    )

    return respuesta.data[0].embedding


def indexar_documentos() -> None:
    archivos = leer_documentos()

    ids = []
    documentos = []
    embeddings = []
    metadatos = []

    total_chunks = 0
    total_tokens = 0

    for archivo in archivos:
        contenido = archivo.read_text(encoding="utf-8")
        chunks = dividir_en_chunks(contenido)

        for chunk_id, chunk in enumerate(chunks):
            tokens_chunk = contar_tokens(chunk)
            embedding = crear_embedding(chunk)

            ids.append(f"{archivo.name}_{chunk_id}")
            documentos.append(chunk)
            embeddings.append(embedding)
            metadatos.append({
                "archivo": archivo.name,
                "chunk_id": chunk_id
            })

            total_chunks += 1
            total_tokens += tokens_chunk

    collection.upsert(
        ids=ids,
        documents=documentos,
        embeddings=embeddings,
        metadatas=metadatos
    )

    coste_estimado = (total_tokens / 1000) * COSTE_POR_1000_TOKENS

    print("Indexación completada")
    print(f"Documentos procesados: {len(archivos)}")
    print(f"Chunks generados: {total_chunks}")
    print(f"Tokens procesados: {total_tokens}")
    print(f"Coste estimado: {coste_estimado:.4f}")


if __name__ == "__main__":
    indexar_documentos()