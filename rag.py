"""
Pipeline RAG per lo Studio Legale Rossi & Partners.
Caricamento documenti, embedding Gemini, storage/retrieval ChromaDB, generazione risposta.
"""

import os
import re
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from google.genai import types

DOCUMENTS_DIR = Path(__file__).parent / "documenti"
COLLECTION_NAME = "studio_legale_docs"

# Modelli Gemini: verificare disponibilità sul free tier (nomi soggetti a aggiornamenti Google).
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.environ.get("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

CHROMA_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))

NO_INFO_MESSAGE = (
    "Non ho trovato questa informazione nei documenti disponibili. "
    "Non è un parere legale: ti consiglio di verificare con lo studio."
)


def _get_gemini_client() -> genai.Client:
    api_key = os.environ["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)


def _get_chroma_client() -> chromadb.HttpClient:
    # ChromaDB gira in container separato: in Compose l'host è il nome del servizio (es. "chromadb").
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[dict[str, str]]:
    """Carica tutti i file .txt dalla cartella documenti."""
    documents: list[dict[str, str]] = []
    for file_path in sorted(documents_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        documents.append({"source": file_path.name, "text": text})
    if not documents:
        raise FileNotFoundError(f"Nessun documento trovato in {documents_dir}")
    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Suddivide il testo in chunk con overlap per preservare il contesto ai confini."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Crea chunk indicizzabili con metadati di provenienza."""
    chunks: list[dict[str, Any]] = []
    for doc in documents:
        for index, chunk in enumerate(chunk_text(doc["text"])):
            chunks.append(
                {
                    "id": f"{doc['source']}_chunk_{index:03d}",
                    "document": chunk,
                    "metadata": {"source": doc["source"], "chunk_index": index},
                }
            )
    return chunks


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    """
    Genera embedding con Gemini.
    task_type RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY migliora la qualità del retrieval
    (supportato da gemini-embedding-001; verificare documentazione per modelli più recenti).
    """
    embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        for emb in response.embeddings:
            embeddings.append(list(emb.values))
    return embeddings


def get_or_create_collection(client: chromadb.HttpClient):
    # Cosine similarity: distanza angolare tra vettori, adatta a embedding normalizzati di Gemini.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(force_reindex: bool = False) -> int:
    """
    Carica documenti, genera embedding e li scrive su ChromaDB.
    Ritorna il numero di chunk indicizzati.
    """
    chroma = _get_chroma_client()
    gemini = _get_gemini_client()

    if force_reindex:
        try:
            chroma.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = get_or_create_collection(chroma)
    if collection.count() > 0 and not force_reindex:
        return collection.count()

    documents = load_documents()
    chunks = build_chunks(documents)
    texts = [c["document"] for c in chunks]
    embeddings = embed_texts(gemini, texts, task_type="RETRIEVAL_DOCUMENT")

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Recupera i top-k chunk più rilevanti da ChromaDB per la query."""
    chroma = _get_chroma_client()
    gemini = _get_gemini_client()
    collection = get_or_create_collection(chroma)

    query_embedding = embed_texts(gemini, [query], task_type="RETRIEVAL_QUERY")[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict[str, Any]] = []
    if not results["documents"] or not results["documents"][0]:
        return retrieved

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "document": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": distance,
            }
        )
    return retrieved


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Fonte {i}: {chunk['source']}]\n{chunk['document']}"
        )
    return "\n\n".join(parts)


def _extract_sources(chunks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        source = chunk["source"]
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def generate_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Costruisce il prompt aumentato e chiama Gemini per la generazione.
    La risposta deve basarsi esclusivamente sui chunk recuperati.
    """
    if not retrieved_chunks:
        return {"answer": NO_INFO_MESSAGE, "sources": []}

    # Soglia euristica: con cosine distance, valori > 0.6 indicano bassa rilevanza.
    relevant = [c for c in retrieved_chunks if c["distance"] < 0.6]
    if not relevant:
        return {"answer": NO_INFO_MESSAGE, "sources": []}

    gemini = _get_gemini_client()
    context = _format_context(relevant)
    sources = _extract_sources(relevant)

    system_instruction = (
        "Sei un assistente dello Studio Legale Rossi & Partners. "
        "Rispondi ESCLUSIVAMENTE in base ai documenti forniti nel contesto. "
        "Se l'informazione richiesta non è presente nei documenti, rispondi esattamente: "
        f'"{NO_INFO_MESSAGE}" '
        "Non inventare informazioni. Rispondi sempre in italiano. "
        "Alla fine della risposta, su una riga separata, indica le fonti nel formato: "
        "Fonte: nomefile.txt (elenca tutti i file usati, separati da virgola se più di uno)."
    )

    prompt = (
        f"Contesto documentale:\n{context}\n\n"
        f"Domanda del cliente: {query}\n\n"
        "Rispondi in modo chiaro e professionale, citando le fonti."
    )

    response = gemini.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )

    answer_text = response.text.strip() if response.text else NO_INFO_MESSAGE

    if NO_INFO_MESSAGE.split(".")[0] in answer_text:
        return {"answer": NO_INFO_MESSAGE, "sources": []}

    if "Fonte:" not in answer_text:
        source_line = "Fonte: " + ", ".join(sources)
        answer_text = f"{answer_text}\n{source_line}"

    return {"answer": answer_text, "sources": sources}


def ask(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Pipeline completa: retrieval + generazione."""
    chunks = retrieve(query, top_k=top_k)
    return generate_answer(query, chunks)
