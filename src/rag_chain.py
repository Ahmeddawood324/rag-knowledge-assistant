"""Step 3 - RAG question-answering pipeline (with memory).

Flow for every user message:

    user question
        |  (1) CONDENSE  - rewrite follow-ups into a standalone question
        v                  using the chat history
    standalone question
        |  (2) RETRIEVE  - FAISS similarity search, top-k chunks
        v
    retrieved context
        |  (3) GENERATE  - LLM answers from context + chat history
        v
    answer  ->  saved to memory
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import Settings
from .indexer import get_embeddings, load_or_build_index
from .llm import UNKNOWN, get_llm
from .memory import BufferMemory, SummaryBufferMemory

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    "Given the chat history and a follow-up question, rewrite the follow-up as ONE "
    "standalone question that includes every name, course code or topic it refers to. "
    "Keep the same language as the follow-up. If it is already standalone, return it "
    "unchanged. Return only the question.\n\n"
    "Chat history:\n{history}\n\n"
    "Follow-up question: {question}\n\n"
    "Standalone question:"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledge assistant for students of Delta Institute of Technology.\n"
            "Rules:\n"
            "1. Answer using ONLY the information in the Context block of the last message.\n"
            f'2. If the answer is not in the context, reply exactly: "{UNKNOWN}"\n'
            "3. Questions about the conversation itself (for example 'what did I ask "
            "before?') are answered from the chat history, not from the context.\n"
            "4. Use the chat history to understand follow-up questions (it, that course, ...).\n"
            "5. Reply in the same language as the user's question.\n"
            "6. Be concise and cite sources like (academic_handbook.pdf, p.3) or "
            "(courses.csv, row 5).\n\n"
            "Summary of earlier conversation: {summary}",
        ),
        MessagesPlaceholder("history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class Answer:
    question: str
    standalone_question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _location(meta: dict) -> str:
    if meta.get("type") == "pdf":
        return f"p.{meta.get('page')}"
    if meta.get("type") == "sheet_summary":
        return "sheet summary"
    return f"row {meta.get('row')}"


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source')}, {_location(d.metadata)})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


class KnowledgeAssistant:
    def __init__(self, vectorstore: FAISS, llm: BaseChatModel, memory: BufferMemory, top_k: int = 4):
        self.vectorstore = vectorstore
        self.memory = memory
        self.top_k = top_k
        self._condense = CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer = ANSWER_PROMPT | llm | StrOutputParser()

    # (1) -------------------------------------------------------------------
    def rewrite_question(self, question: str) -> str:
        if self.memory.turn_count == 0:
            return question
        rewritten = self._condense.invoke(
            {"history": self.memory.history_text(), "question": question}
        )
        return rewritten.strip().strip('"') or question

    # (2) -------------------------------------------------------------------
    def retrieve(self, query: str) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=self.top_k)

    # (3) -------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        standalone = self.rewrite_question(question)
        docs = self.retrieve(standalone)
        answer = self._answer.invoke(
            {
                "summary": self.memory.get_summary() or "(none)",
                "history": self.memory.get_messages(),
                "context": format_context(docs),
                "question": question,
            }
        ).strip()
        self.memory.add_turn(question, answer)  # <- the assistant "remembers"
        return Answer(
            question=question,
            standalone_question=standalone,
            answer=answer,
            sources=[
                {"source": d.metadata.get("source"), "location": _location(d.metadata)}
                for d in docs
            ],
            contexts=[d.page_content for d in docs],
            seconds=time.perf_counter() - start,
        )

    def reset(self) -> None:
        self.memory.clear()


def build_assistant(
    settings: Settings | None = None, force_rebuild: bool = False
) -> tuple[KnowledgeAssistant, dict]:
    """Wire everything together: load data -> index -> LLM -> memory -> assistant."""
    settings = settings or Settings.from_env()
    store, info = load_or_build_index(settings, get_embeddings(settings), force_rebuild)
    llm = get_llm(settings)
    memory = (
        SummaryBufferMemory(llm, settings.summary_max_turns)
        if settings.memory_mode == "summary_buffer"
        else BufferMemory()
    )
    return KnowledgeAssistant(store, llm, memory, settings.top_k), info
