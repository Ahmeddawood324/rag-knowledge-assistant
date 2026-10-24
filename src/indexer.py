"""Step 2 - Embedding & indexing (FAISS)."""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .loaders import load_all


# --------------------------------------------------------------------------
# Embedding models
# --------------------------------------------------------------------------
class HashingEmbeddings(Embeddings):
    """Tiny offline bag-of-words embedding (no download, no API key).

    ONLY for tests / demos without internet. It matches shared words, it does
    not understand meaning - use HuggingFace or OpenAI embeddings for real use.
    """

    STOPWORDS = frozenset(
        "the a an is are was were of to in on for and or what which who whom how many much do does "
        "did i you me my it its be can will about tell with from that this there their when where why "
        "at as by if not no".split()
    )

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            if tok in self.STOPWORDS:
                continue
            vec[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        vec = np.sqrt(vec)  # dampen repeated words
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)
    if provider == "fake":
        return HashingEmbeddings()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    """PDF pages are split into overlapping chunks; sheet rows are already
    small, self-contained facts, so they are kept whole."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("type") == "pdf":
            chunks += splitter.split_documents([doc])
        else:
            chunks.append(doc)
    return chunks


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
def _fingerprint(chunks: list[Document], settings: Settings) -> str:
    h = hashlib.sha256()
    h.update(f"{settings.embedding_provider}|{settings.embedding_model}".encode())
    h.update(f"{settings.chunk_size}|{settings.chunk_overlap}".encode())
    for c in chunks:
        h.update(c.page_content.encode("utf-8"))
    return h.hexdigest()


def load_or_build_index(
    settings: Settings, embeddings: Embeddings, force_rebuild: bool = False
) -> tuple[FAISS, dict]:
    """Return (vector_store, info). The index is cached on disk and rebuilt
    automatically when the data, chunking or embedding model changes."""
    chunks = split_documents(load_all(settings), settings)
    if not chunks:
        raise RuntimeError(
            f"No documents found. Put PDFs in {settings.pdf_dir} and CSVs in "
            f"{settings.sheets_dir} (or set SHEET_URLS)."
        )
    fingerprint = _fingerprint(chunks, settings)
    manifest_path = settings.index_dir / "manifest.json"

    if not force_rebuild and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") == fingerprint:
            store = FAISS.load_local(
                str(settings.index_dir), embeddings, allow_dangerous_deserialization=True
            )
            return store, {"chunks": manifest["chunks"], "rebuilt": False}

    store = FAISS.from_documents(chunks, embeddings)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.index_dir))
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "chunks": len(chunks)}))
    return store, {"chunks": len(chunks), "rebuilt": True}
