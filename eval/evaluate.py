"""Step 5 - Testing & evaluation.

    python -m eval.evaluate --offline          # no API key needed (mock LLM)
    python -m eval.evaluate                    # real LLM from .env
    python -m eval.evaluate --judge            # + LLM-as-judge relevance score (real LLM only)

Metrics
-------
retrieval_hit     did the retrieved chunks come from the expected file?
context_recall    do the retrieved chunks contain the expected keywords?
answer_accuracy   share of expected keywords that appear in the final answer
answer_pass       1 if ALL expected keywords appear (strict)
refusal_ok        for unanswerable questions: did the assistant admit it does not know?
judge_relevance   (optional) 1-5 score given by an LLM for relevance to the question
Conversations are run with a fresh memory, turn by turn, so follow-up and
memory behaviour is tested too.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from src.config import Settings  # noqa: E402
from src.rag_chain import build_assistant  # noqa: E402

HERE = Path(__file__).resolve().parent
REFUSAL_MARKERS = ["don't know", "do not know", "not in the", "no information", "cannot find",
                   "can't find", "not mentioned", "لا أعرف", "لا توجد", "غير موجود"]

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are grading an assistant's answer.\nQuestion: {question}\nAnswer: {answer}\n"
    "Rate how relevant and helpful the answer is to the question from 1 (useless) to 5 "
    "(perfect). Reply with the number only.\nScore:"
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def keyword_share(keywords: list[str], text: str) -> float:
    if not keywords:
        return 1.0
    t = norm(text)
    return sum(norm(k) in t for k in keywords) / len(keywords)


def run(args: argparse.Namespace) -> None:
    overrides = {"index_dir": Path(tempfile.mkdtemp()) / "index"}
    if args.offline:
        overrides.update(llm_provider="mock", embedding_provider="fake", llm_model="", embedding_model="")
    settings = Settings.from_env(**overrides)
    semantic = settings.embedding_provider != "fake"

    assistant, info = build_assistant(settings, force_rebuild=True)
    judge_chain = None
    if args.judge and settings.llm_provider != "mock":
        from src.llm import get_llm
        judge_chain = JUDGE_PROMPT | get_llm(settings)

    cases = json.loads((HERE / "test_cases.json").read_text(encoding="utf-8"))["cases"]
    rows, skipped = [], []
    for case in cases:
        if case.get("requires_semantic") and not semantic:
            skipped.append(case["id"])
            continue
        assistant.reset()  # new chat session per case
        for turn_no, turn in enumerate(case["turns"], start=1):
            res = assistant.ask(turn["question"])
            kws = turn.get("expected_keywords", [])
            srcs = turn.get("expected_sources")
            row = {
                "case": case["id"], "category": case["category"], "turn": turn_no,
                "question": turn["question"], "standalone": res.standalone_question,
                "answer": res.answer, "seconds": round(res.seconds, 2),
                "retrieval_hit": None if not srcs else float(any(s["source"] in srcs for s in res.sources)),
                "context_recall": None if (not kws or not srcs) else keyword_share(kws, " ".join(res.contexts)),
                "answer_accuracy": None, "answer_pass": None, "refusal_ok": None, "judge_relevance": None,
            }
            if turn.get("expect_unanswerable"):
                row["refusal_ok"] = float(any(m in res.answer.lower() for m in REFUSAL_MARKERS))
            else:
                row["answer_accuracy"] = keyword_share(kws, res.answer)
                row["answer_pass"] = float(row["answer_accuracy"] == 1.0)
            if judge_chain:
                raw = judge_chain.invoke({"question": turn["question"], "answer": res.answer}).content
                m = re.search(r"[1-5]", str(raw))
                row["judge_relevance"] = int(m.group()) if m else None
            rows.append(row)

    write_report(rows, skipped, settings, info)


def mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(vals) if vals else None


def fmt(v) -> str:
    return "-" if v is None else f"{v:.0%}" if v <= 1 else f"{v:.2f}"


def write_report(rows, skipped, settings, info) -> None:
    metrics = ["retrieval_hit", "context_recall", "answer_accuracy", "answer_pass", "refusal_ok"]
    if any(r["judge_relevance"] is not None for r in rows):
        metrics.append("judge_relevance")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    lines = [
        "# Evaluation report", "",
        f"- LLM: `{settings.llm_provider}:{settings.llm_model}`  |  embeddings: `{settings.embedding_provider}`"
        f"  |  chunks: {info['chunks']}  |  top_k: {settings.top_k}",
        f"- Questions asked: {len(rows)}  |  skipped (need semantic embeddings): {', '.join(skipped) or 'none'}",
        f"- Average latency per question: {statistics.mean(r['seconds'] for r in rows):.2f}s", "",
        "## Scores by category", "",
        "| category | n | " + " | ".join(metrics) + " |",
        "|---|---|" + "---|" * len(metrics),
    ]
    for cat, rs in by_cat.items():
        lines.append(f"| {cat} | {len(rs)} | " + " | ".join(fmt(mean(rs, m)) for m in metrics) + " |")
    lines.append("| **all** | %d | " % len(rows) + " | ".join(fmt(mean(rows, m)) for m in metrics) + " |")
    lines += ["", "## Per-question details", "",
              "| case | turn | question | rewritten for search | answer | acc | retr |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ans = r["answer"].replace("|", "/").replace("\n", " ")[:140]
        acc = r["answer_accuracy"] if r["answer_accuracy"] is not None else r["refusal_ok"]
        lines.append(
            f"| {r['case']} | {r['turn']} | {r['question']} | "
            f"{r['standalone'][:80] if r['standalone'] != r['question'] else '(same)'} | {ans} | "
            f"{fmt(acc)} | {fmt(r['retrieval_hit'])} |"
        )
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines[:14 + len(by_cat)]))
    print(f"\nFull report: {out / 'report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="mock LLM + hashing embeddings")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge relevance score")
    run(ap.parse_args())
