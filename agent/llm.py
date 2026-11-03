"""Tool-calling LLM backends behind one tiny interface: chat(messages, tools) -> Turn.

The model never runs a tool itself. It only *asks* for one by emitting a structured call; our loop runs it and
feeds the result back. Ollama parses those calls for us; with a raw Hugging Face model we parse Qwen3's
`<tool_call>{...}</tool_call>` tags ourselves, which shows what "function calling" really is: text in a format.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: Literal["ollama", "transformers"] = "ollama"
    llm_model: str = "qwen3:4b"
    ollama_url: str = "http://localhost:11434"
    llm_max_tokens: int = 512


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Turn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)  # malformed calls: counted in the eval


class ChatLLM(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Turn: ...


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> Turn:
    calls, errors = [], []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", {})
            calls.append(ToolCall(obj["name"], json.loads(args) if isinstance(args, str) else args))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errors.append(f"{type(e).__name__}: {raw[:200]}")
    content = TOOL_CALL_RE.sub("", text)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return Turn(content, calls, errors)


class OllamaChat:
    def __init__(self, s: Settings):
        self.s, self.name = s, f"ollama:{s.llm_model}"

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Turn:
        body = {"model": self.s.llm_model, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 0, "num_predict": self.s.llm_max_tokens}}  # fmt: skip
        if tools:
            body["tools"] = tools
        r = httpx.post(f"{self.s.ollama_url}/api/chat", json=body, timeout=300)
        r.raise_for_status()
        msg = r.json()["message"]
        calls = [
            ToolCall(c["function"]["name"], c["function"].get("arguments") or {}) for c in msg.get("tool_calls") or []
        ]
        return Turn(msg.get("content", "").strip(), calls)


class TransformersChat:
    def __init__(self, s: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.s, self.name = s, f"transformers:{s.llm_model}"
        torch.manual_seed(0)
        self.tok = AutoTokenizer.from_pretrained(s.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(s.llm_model, torch_dtype="auto").eval()

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Turn:
        import torch

        kw = {"enable_thinking": False} if "qwen3" in self.s.llm_model.lower() else {}
        enc = self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kw
        )
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.s.llm_max_tokens, do_sample=False)
        # keep special tokens: <tool_call> must survive decoding; strip only the end-of-turn markers
        text = self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=False)
        return parse_tool_calls(re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text))


def get_chat(s: Settings | None = None) -> ChatLLM:
    s = s or Settings()
    return OllamaChat(s) if s.llm_provider == "ollama" else TransformersChat(s)
