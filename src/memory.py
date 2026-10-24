"""Step 4 - Conversation memory.

Two strategies (same interface):

* BufferMemory         - keeps the FULL chat history (required by the task).
* SummaryBufferMemory  - keeps the last N turns verbatim and compresses older
                         turns into a running summary (optional, for long chats).

They are written with plain `langchain_core` messages instead of the old
`ConversationBufferMemory` classes, which were deprecated/removed in recent
LangChain versions. Same idea, no dependency on removed APIs.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    "Progressively summarize the conversation between a student and an assistant. "
    "Keep names, numbers, course codes and the topics discussed. "
    "Write the summary in the language of the conversation, max 120 words.\n\n"
    "Current summary:\n{summary}\n\n"
    "New lines of conversation:\n{new_lines}\n\n"
    "New summary:"
)


class BufferMemory:
    """Stores every turn. Simple and exact, but grows without limit."""

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    # --- write -------------------------------------------------------------
    def add_turn(self, user: str, ai: str) -> None:
        self._messages += [HumanMessage(content=user), AIMessage(content=ai)]

    def clear(self) -> None:
        self._messages = []

    # --- read --------------------------------------------------------------
    @property
    def turn_count(self) -> int:
        return len(self._messages) // 2

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def get_summary(self) -> str:
        return ""

    def history_text(self, max_chars_per_message: int = 600) -> str:
        lines = []
        for m in self._messages:
            who = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{who}: {str(m.content)[:max_chars_per_message]}")
        return "\n".join(lines)

    # --- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        data = [{"role": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                for m in self._messages]
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        self.clear()
        for item in json.loads(Path(path).read_text(encoding="utf-8")):
            cls = HumanMessage if item["role"] == "user" else AIMessage
            self._messages.append(cls(content=item["content"]))


class SummaryBufferMemory(BufferMemory):
    """Recent turns verbatim + running summary of everything older."""

    def __init__(self, llm: BaseChatModel, max_turns: int = 6) -> None:
        super().__init__()
        self.max_turns = max_turns
        self._summary = ""
        self._chain = SUMMARIZE_PROMPT | llm | StrOutputParser()

    def add_turn(self, user: str, ai: str) -> None:
        super().add_turn(user, ai)
        overflow = self.turn_count - self.max_turns
        if overflow > 0:
            old, self._messages = self._messages[: overflow * 2], self._messages[overflow * 2:]
            new_lines = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {str(m.content)[:600]}"
                for m in old
            )
            self._summary = self._chain.invoke(
                {"summary": self._summary or "(empty)", "new_lines": new_lines}
            ).strip()

    def clear(self) -> None:
        super().clear()
        self._summary = ""

    def get_summary(self) -> str:
        return self._summary

    def history_text(self, max_chars_per_message: int = 600) -> str:
        recent = super().history_text(max_chars_per_message)
        if self._summary:
            return f"Summary of earlier conversation: {self._summary}\n{recent}"
        return recent
