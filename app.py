"""Command-line chat with the knowledge assistant.

    python app.py                 # normal run (uses .env)
    python app.py --rebuild       # force re-indexing of PDFs / Sheets
    python app.py --offline       # no API key / no downloads (mock LLM + hashing embeddings)
    python app.py --memory summary_buffer

In-chat commands:  /history  /sources  /reset  /exit
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from src.config import Settings  # noqa: E402
from src.rag_chain import build_assistant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the FAISS index")
    parser.add_argument("--offline", action="store_true", help="mock LLM + hashing embeddings")
    parser.add_argument("--memory", choices=["buffer", "summary_buffer"], help="memory type")
    args = parser.parse_args()

    overrides = {}
    if args.offline:
        overrides.update(llm_provider="mock", embedding_provider="fake", llm_model="", embedding_model="")
    if args.memory:
        overrides["memory_mode"] = args.memory
    settings = Settings.from_env(**overrides)

    print("Loading knowledge base ...")
    assistant, info = build_assistant(settings, force_rebuild=args.rebuild)
    print(
        f"Ready: {info['chunks']} chunks indexed ({'built now' if info['rebuilt'] else 'loaded from cache'}) | "
        f"LLM={settings.llm_provider}:{settings.llm_model} | "
        f"embeddings={settings.embedding_provider} | memory={settings.memory_mode}"
    )
    print("Ask me anything about the handbook, services, courses or fees. /exit to quit.\n")

    last = None
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/reset":
            assistant.reset()
            print("(memory cleared)\n")
            continue
        if question == "/history":
            print(assistant.memory.history_text() or "(empty)", "\n")
            continue
        if question == "/sources":
            if last:
                for s in last.sources:
                    print(f"  - {s['source']} ({s['location']})")
                print(f"  standalone question used for search: {last.standalone_question}\n")
            continue

        last = assistant.ask(question)
        print(f"\nAssistant: {last.answer}")
        print("Sources: " + ", ".join(sorted({f"{s['source']} ({s['location']})" for s in last.sources})))
        print()


if __name__ == "__main__":
    main()
