# Delta Assistant - AI Knowledge Assistant with Memory (RAG + LangChain)

Mid-term project (Week 3). A chat assistant that answers questions from **PDFs** and **Google Sheets**,
cites its sources, and **remembers the conversation** so follow-up questions work
("Tell me about Machine Learning" -> "What are its prerequisites?").

The demo knowledge base is a fictional university (handbook, services guide, course catalog, fees).
Replace the files in `data/` with your own to use any other domain.

## Architecture

```
 PDFs ----------\                                  +--> chat memory (Buffer / SummaryBuffer)
                 >-> text Documents -> chunks -> embeddings -> FAISS
 Google Sheets -/                                        ^
                                                         | top-k similar chunks
 user question -> (1) rewrite follow-up using history -> (2) retrieve -> (3) LLM answer + sources
```

| Step | File | What it does |
|---|---|---|
| 1. Data integration | `src/loaders.py` | PDF pages -> Documents; sheet rows -> `Column: value` text |
| 2. Embedding & indexing | `src/indexer.py` | chunking, embeddings (HuggingFace / OpenAI), FAISS index cached on disk |
| 3. QA pipeline | `src/rag_chain.py` | condense -> retrieve -> generate, returns answer + sources |
| 4. Memory | `src/memory.py` | `BufferMemory` (full history), `SummaryBufferMemory` (optional) |
| 5. Evaluation | `eval/evaluate.py`, `tests/` | retrieval/answer metrics + unit tests |
| LLM factory | `src/llm.py` | OpenAI / Gemini / Ollama / offline mock |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your key in .env
python app.py                                          # chat
```

No API key yet? Try the whole pipeline offline (fake LLM + hashing embeddings - plumbing only, not real AI):

```bash
python app.py --offline
python -m eval.evaluate --offline
pytest -q
```

Free options: `LLM_PROVIDER=gemini` (free tier) or `ollama` (local); embeddings `huggingface` are free and local.

### Use your own Google Sheets
1. Share each sheet: *Share -> Anyone with the link -> Viewer*.
2. Put the links in `.env`: `SHEET_URLS=courses=https://docs.google.com/spreadsheets/d/XXXX/edit?gid=0`
3. Run `python app.py --rebuild`.

The code converts the link to the CSV export URL; no Google API/OAuth needed. If a sheet cannot be
loaded a warning is printed and the other sources still work.

### Use your own PDFs
Drop them in `data/pdfs/` and run `python app.py --rebuild` (the index also rebuilds automatically when data changes).
Scanned PDFs (images) need OCR first - `pypdf` only reads real text.

## Chat commands
`/history` show memory - `/sources` show retrieved chunks and the rewritten search query - `/reset` clear memory - `/exit`

## Memory modes
`MEMORY_MODE=buffer` (default) keeps the full history. `MEMORY_MODE=summary_buffer` keeps the last
`SUMMARY_MAX_TURNS` turns and compresses older ones into a summary using the LLM - cheaper for long chats.

## Evaluation
`eval/test_cases.json` has 24 questions (single, follow-up, memory, unanswerable, Arabic).
`python -m eval.evaluate` writes `eval/results/report.md` with retrieval hit, context recall,
answer accuracy, refusal correctness, latency, and (with `--judge`) an LLM relevance score.

Important: the report shipped in `eval/results/` was produced with the **offline mock** (it proves the
pipeline, retrieval and memory logic work, but it does not measure real LLM quality). Run it with your
real LLM/embeddings and use those numbers in your presentation.

## Known limitations (good to mention in the discussion)
- Aggregation questions ("how many courses are in level 3?") are weak in RAG: only top-k rows are seen.
- Tables inside PDFs may extract as messy text; scanned PDFs need OCR.
- `top_k=4` can miss answers spread over many chunks; tune `TOP_K` / `CHUNK_SIZE`.
- Buffer memory grows forever; use `summary_buffer` for long sessions.
- The FAISS index is loaded with `allow_dangerous_deserialization=True`; only load indexes you built yourself.
