"""Central configuration. Everything can be changed from the .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1",
    "mock": "mock",
}
DEFAULT_EMBEDDING_MODELS = {
    # multilingual -> understands Arabic + English questions
    "huggingface": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "openai": "text-embedding-3-small",
    "fake": "hashing-bow-4096",
}


@dataclass
class Settings:
    # --- models -----------------------------------------------------------
    llm_provider: str = "openai"          # openai | gemini | ollama | mock
    llm_model: str = ""
    embedding_provider: str = "huggingface"  # huggingface | openai | fake
    embedding_model: str = ""

    # --- chunking / retrieval --------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- memory -------------------------------------------------------------
    memory_mode: str = "buffer"           # buffer | summary_buffer
    summary_max_turns: int = 6            # recent turns kept verbatim (summary_buffer)

    # --- data sources -----------------------------------------------------
    pdf_dir: Path = ROOT / "data" / "pdfs"
    sheets_dir: Path = ROOT / "data" / "sheets"
    # Google Sheets links, optionally "name=url". Empty -> use local CSV exports.
    sheet_urls: list[str] = field(default_factory=list)

    # --- storage ------------------------------------------------------------
    index_dir: Path = ROOT / "storage" / "faiss_index"

    def __post_init__(self) -> None:
        self.pdf_dir = Path(self.pdf_dir)
        self.sheets_dir = Path(self.sheets_dir)
        self.index_dir = Path(self.index_dir)
        if not self.llm_model:
            self.llm_model = DEFAULT_LLM_MODELS.get(self.llm_provider, "")
        if not self.embedding_model:
            self.embedding_model = DEFAULT_EMBEDDING_MODELS.get(self.embedding_provider, "")

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        load_dotenv(ROOT / ".env")
        env = os.environ.get
        urls = [u.strip() for u in env("SHEET_URLS", "").split(",") if u.strip()]
        values = dict(
            llm_provider=env("LLM_PROVIDER", "openai").lower(),
            llm_model=env("LLM_MODEL", ""),
            embedding_provider=env("EMBEDDING_PROVIDER", "huggingface").lower(),
            embedding_model=env("EMBEDDING_MODEL", ""),
            chunk_size=int(env("CHUNK_SIZE", "800")),
            chunk_overlap=int(env("CHUNK_OVERLAP", "120")),
            top_k=int(env("TOP_K", "4")),
            memory_mode=env("MEMORY_MODE", "buffer").lower(),
            summary_max_turns=int(env("SUMMARY_MAX_TURNS", "6")),
            sheet_urls=urls,
        )
        values.update(overrides)
        return cls(**values)
