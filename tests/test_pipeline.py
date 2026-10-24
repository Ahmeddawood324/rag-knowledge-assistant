"""Offline tests: no API key, no internet. Run with:  pytest -q"""
import pytest

from src.config import Settings
from src.indexer import get_embeddings, load_or_build_index, split_documents
from src.llm import UNKNOWN, MockChatModel
from src.loaders import dataframe_to_documents, google_sheet_csv_url, load_all, load_pdfs
from src.memory import BufferMemory, SummaryBufferMemory
from src.rag_chain import build_assistant


@pytest.fixture()
def settings(tmp_path):
    return Settings.from_env(
        llm_provider="mock", embedding_provider="fake", llm_model="", embedding_model="",
        index_dir=tmp_path / "index", sheet_urls=[],
    )


# ---- data integration -----------------------------------------------------
def test_pdf_pages_loaded_with_metadata(settings):
    docs = load_pdfs(settings.pdf_dir)
    assert docs and all(d.metadata["type"] == "pdf" and d.metadata["page"] >= 1 for d in docs)
    assert {d.metadata["source"] for d in docs} == {"academic_handbook.pdf", "student_services.pdf"}


def test_sheet_rows_become_text(settings):
    docs = load_all(settings)
    rows = [d for d in docs if d.metadata["type"] == "sheet"]
    ml = next(d for d in rows if "AI350" in d.page_content)
    assert "Course Name: Machine Learning" in ml.page_content and "Instructor: Dr. Sara Khalil" in ml.page_content
    assert ml.metadata["row"] >= 2


def test_google_sheet_url_conversion():
    url = "https://docs.google.com/spreadsheets/d/ABC123_-x/edit?gid=42#gid=42"
    assert google_sheet_csv_url(url) == "https://docs.google.com/spreadsheets/d/ABC123_-x/export?format=csv&gid=42"
    with pytest.raises(ValueError):
        google_sheet_csv_url("https://example.com/not-a-sheet")


def test_empty_rows_ignored():
    import pandas as pd
    df = pd.DataFrame({"a": ["x", None], "b": ["y", None]})
    assert len(dataframe_to_documents(df, "t")) == 2  # summary + 1 row


# ---- indexing -------------------------------------------------------------
def test_chunking_keeps_rows_whole_and_splits_pdf(settings):
    docs = load_all(settings)
    chunks = split_documents(docs, settings)
    assert all(len(c.page_content) <= settings.chunk_size + 50 for c in chunks if c.metadata["type"] == "pdf")
    assert sum(c.metadata["type"] == "sheet" for c in chunks) == sum(d.metadata["type"] == "sheet" for d in docs)


def test_index_is_cached_and_searchable(settings):
    emb = get_embeddings(settings)
    store, info = load_or_build_index(settings, emb)
    assert info["rebuilt"] is True
    _, info2 = load_or_build_index(settings, emb)
    assert info2["rebuilt"] is False  # loaded from disk
    hit = store.similarity_search("Who teaches Machine Learning?", k=2)
    assert any("AI350" in d.page_content for d in hit)


# ---- memory ---------------------------------------------------------------
def test_buffer_memory_keeps_everything(tmp_path):
    m = BufferMemory()
    for i in range(10):
        m.add_turn(f"q{i}", f"a{i}")
    assert m.turn_count == 10 and m.get_messages()[0].content == "q0"
    m.save(tmp_path / "s.json")
    m2 = BufferMemory()
    m2.load(tmp_path / "s.json")
    assert m2.history_text() == m.history_text()
    m.clear()
    assert m.turn_count == 0


def test_summary_buffer_compresses_old_turns():
    m = SummaryBufferMemory(MockChatModel(), max_turns=2)
    for i in range(5):
        m.add_turn(f"question number {i}", f"answer {i}")
    assert m.turn_count == 2                       # only recent turns verbatim
    assert "question number 0" in m.get_summary()  # old turns survive in summary
    assert "question number 4" not in m.get_summary()


# ---- full pipeline --------------------------------------------------------
def test_answer_comes_with_sources(settings):
    assistant, _ = build_assistant(settings)
    res = assistant.ask("How many credit hours are required to graduate?")
    assert "132" in res.answer
    assert any(s["source"] == "academic_handbook.pdf" for s in res.sources)


def test_followup_is_rewritten_and_answered(settings):
    assistant, _ = build_assistant(settings)
    assistant.ask("Tell me about the Machine Learning course.")
    res = assistant.ask("What are its prerequisites?")
    assert "machine learning" in res.standalone_question.lower()  # context carried over
    assert "AI301" in res.answer


def test_assistant_remembers_earlier_questions(settings):
    assistant, _ = build_assistant(settings)
    assistant.ask("How many credit hours do I need to graduate?")
    assistant.ask("And what CGPA do I need?")
    res = assistant.ask("What did I ask you first?")
    assert "credit hours" in res.answer
    assert assistant.memory.turn_count == 3
    assistant.reset()
    assert assistant.memory.turn_count == 0


def test_unknown_question_is_refused(settings):
    assistant, _ = build_assistant(settings)
    assert assistant.ask("Who won the football World Cup in 2022?").answer == UNKNOWN
