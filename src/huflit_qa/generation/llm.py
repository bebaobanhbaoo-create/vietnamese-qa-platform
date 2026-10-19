"""Chat LLM clients. Ollama runs the model locally; FakeLLM is for tests."""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import httpx

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


@dataclass
class Completion:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: float | None = None


class LLM(Protocol):
    model: str

    def complete(self, messages: list[Message]) -> Completion: ...

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


class LLMUnavailableError(RuntimeError):
    pass


class OllamaLLM:
    """Client for Ollama's /api/chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 512,
        num_ctx: int = 4096,
        timeout_s: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx}
        self.timeout = httpx.Timeout(timeout_s, connect=5.0)

    def _payload(self, messages: list[Message], stream: bool) -> dict:
        return {"model": self.model, "messages": messages, "stream": stream, "options": self.options}

    def _unavailable(self, exc: Exception) -> LLMUnavailableError:
        return LLMUnavailableError(
            f"Cannot reach Ollama at {self.base_url} ({exc}). Install it from https://ollama.com, "
            f"then run `ollama pull {self.model}`."
        )

    def complete(self, messages: list[Message]) -> Completion:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat", json=self._payload(messages, False), timeout=self.timeout
            )
        except httpx.ConnectError as exc:
            raise self._unavailable(exc) from exc
        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Model {self.model} not found in Ollama: run `ollama pull {self.model}`."
            )
        resp.raise_for_status()
        data = resp.json()
        return Completion(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            duration_ms=data["total_duration"] / 1e6 if data.get("total_duration") else None,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/api/chat", json=self._payload(messages, True), timeout=self.timeout
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("message", {}).get("content"):
                        yield part["message"]["content"]
                    if part.get("done"):
                        return
        except httpx.ConnectError as exc:
            raise self._unavailable(exc) from exc


@dataclass
class FakeLLM:
    """Returns canned answers; records the prompts it was given."""

    answer: str = "Theo quy định [1]."
    model: str = "fake"
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: list[Message]) -> Completion:
        self.calls.append(messages)
        return Completion(text=self.answer, prompt_tokens=0, completion_tokens=0, duration_ms=0.0)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append(messages)
        words = self.answer.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
