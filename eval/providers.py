"""Provider adapter layer for multi-model evaluation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


Message = Dict[str, str]


@dataclass
class ProviderResponse:
    text: str
    raw: Dict[str, Any]


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        raise NotImplementedError


def _safe_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # pydantic v2 style
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "openai package is required for OpenAI/Qwen/Llama providers."
            ) from exc
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content or ""
        return ProviderResponse(text=text, raw=_safe_dict(completion))


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("anthropic package is required for anthropic provider.") from exc
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        system = ""
        anthropic_messages: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system = f"{system}\n{msg.get('content', '')}".strip()
            elif role in {"user", "assistant"}:
                anthropic_messages.append({"role": role, "content": msg.get("content", "")})
        resp = self.client.messages.create(
            model=model,
            messages=anthropic_messages,
            system=system or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text_chunks = []
        for block in resp.content:
            if getattr(block, "type", "") == "text":
                text_chunks.append(getattr(block, "text", ""))
        text = "\n".join(text_chunks).strip()
        return ProviderResponse(text=text, raw=_safe_dict(resp))


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderError(
                "google-generativeai package is required for google provider."
            ) from exc
        self.genai = genai
        self.genai.configure(api_key=api_key)

    def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        # Gemini's API shape differs from role-based chat APIs, so we flatten
        # transcript context into one prompt while preserving turn order.
        transcript = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            transcript.append(f"{role}: {msg.get('content', '')}")
        prompt = "\n\n".join(transcript)

        generation_config = {"temperature": temperature, "max_output_tokens": max_tokens}
        model_obj = self.genai.GenerativeModel(model_name=model)
        resp = model_obj.generate_content(prompt, generation_config=generation_config)
        text = getattr(resp, "text", "") or ""
        return ProviderResponse(text=text, raw=_safe_dict(resp))


def create_provider(name: str, extra_config: Optional[Dict[str, Any]] = None) -> BaseProvider:
    cfg = extra_config or {}
    provider = name.lower().strip()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for provider=openai.")
        return OpenAIProvider(api_key=api_key)

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for provider=anthropic.")
        return AnthropicProvider(api_key=api_key)

    if provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError("GOOGLE_API_KEY is required for provider=google.")
        return GoogleProvider(api_key=api_key)

    if provider == "qwen":
        api_key = os.environ.get("QWEN_API_KEY", "dummy")
        base_url = cfg.get("qwen_base_url") or os.environ.get("QWEN_BASE_URL")
        if not base_url:
            raise ProviderError(
                "QWEN_BASE_URL (or --qwen_base_url) is required for provider=qwen."
            )
        return OpenAIProvider(api_key=api_key, base_url=base_url)

    if provider == "llama":
        api_key = os.environ.get("LLAMA_API_KEY", "dummy")
        base_url = cfg.get("llama_base_url") or os.environ.get("LLAMA_BASE_URL")
        if not base_url:
            raise ProviderError(
                "LLAMA_BASE_URL (or --llama_base_url) is required for provider=llama."
            )
        return OpenAIProvider(api_key=api_key, base_url=base_url)

    raise ProviderError(
        f"Unsupported provider '{name}'. Use one of: "
        "openai, anthropic, google, qwen, llama."
    )


def provider_metadata(name: str, model: str, extra_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base: Dict[str, Any] = {"provider": name, "model": model}
    if not extra_config:
        return base
    # Only include useful non-secret metadata in logs.
    for k in ("qwen_base_url", "llama_base_url"):
        if extra_config.get(k):
            base[k] = extra_config[k]
    return base

