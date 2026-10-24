"""LLM factory + an offline MockChatModel used for tests and demos."""
from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import Settings

UNKNOWN = "I don't know based on the provided documents."


def get_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model, temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.llm_model, temperature=0)
    if provider == "mock":
        return MockChatModel()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


# --------------------------------------------------------------------------
# Offline mock (NOT an AI - a deterministic stand-in so the whole pipeline
# can be tested without an API key). It recognises the three prompts used in
# this project and answers with simple rules.
# --------------------------------------------------------------------------
_STOP = set(
    "the a an is are was were of to in on for and or what which who whom how many much "
    "do does did i you me my it its be can will about tell with from that this there their "
    "when where why at as by if not no yes need needs should would could".split()
)
_PRONOUNS = {"it", "its", "that", "this", "they", "them", "their", "those", "these", "he",
             "she", "his", "her", "also", "and", "too", "same"}
_RECALL = re.compile(r"what did i (ask|say)|my first question|first question", re.I)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP and len(t) > 1]


class MockChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = str(messages[-1].content)
        if text.rstrip().endswith("Standalone question:"):
            out = self._condense(text)
        elif text.rstrip().endswith("New summary:"):
            out = self._summarize(text)
        elif "Context:" in text and "Question:" in text:
            out = self._answer(messages)
        else:
            out = "3"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])

    # -- the three behaviours -------------------------------------------------
    @staticmethod
    def _condense(text: str) -> str:
        history = re.search(r"Chat history:\n(.*)\n\nFollow-up question:", text, re.S).group(1)
        question = re.search(r"Follow-up question: (.*)\n\nStandalone question:", text, re.S).group(1)
        words = re.findall(r"\w+", question.lower())
        if len(words) <= 7 or _PRONOUNS & set(words):
            earlier = [l[6:] for l in history.splitlines() if l.startswith("User: ")]
            return " ".join(earlier + [question]).strip()
        return question

    @staticmethod
    def _summarize(text: str) -> str:
        old = re.search(r"Current summary:\n(.*?)\n\nNew lines", text, re.S).group(1)
        users = re.findall(r"^User: (.*)$", text, re.M)
        base = "" if old.strip() == "(empty)" else old.strip() + " "
        return (base + "The user asked: " + "; ".join(users))[:600]

    @staticmethod
    def _answer(messages: list[BaseMessage]) -> str:
        last = str(messages[-1].content)
        question = last.split("Question:", 1)[1].strip()
        history_users = [str(m.content) for m in messages[1:-1] if isinstance(m, HumanMessage)]

        if _RECALL.search(question) and history_users:
            return f"Your first question was: {history_users[0]}"

        context = last.split("Context:", 1)[1].split("\n\nQuestion:", 1)[0]
        blocks = re.split(r"\n\n(?=\[\d+\] \()", context.strip())
        q_tokens = set(_tokens(question))
        q_words = re.findall(r"\w+", question.lower())
        if history_users and (len(q_words) <= 7 or _PRONOUNS & set(q_words)):
            # follow-up question: borrow the topic words from earlier user turns,
            # exactly what a real LLM does by reading the chat history
            for earlier in history_users:
                q_tokens |= set(_tokens(earlier))
        best, best_score = None, 0
        for block in blocks:
            score = len(q_tokens & set(_tokens(block)))
            if score > best_score:        # ties -> keep earlier (= better ranked) block
                best, best_score = block, score
        if best is None or best_score < 2:
            return UNKNOWN
        header, _, body = best.partition("\n")
        # for long passages keep only the best matching sentences
        sentences = re.split(r"(?<=[.!?])\s+", body)
        if len(sentences) > 2:
            ranked = sorted(sentences, key=lambda s: -len(q_tokens & set(_tokens(s))))[:2]
            body = " ".join(s for s in sentences if s in ranked)
        source = re.search(r"\((.*?)\)", header).group(1)
        return f"{body.strip()} ({source})"
