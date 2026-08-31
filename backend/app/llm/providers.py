import base64
import os

import httpx

from app.llm.base import LlmProvider, LlmProviderError, classify_http_error


class GeminiProvider(LlmProvider):
    """Google AI Studio's free-tier Gemini API. Chosen as the default first
    link in the chain because it's the one free provider that reliably does
    both function-calling-grade structured JSON *and* image input in the same
    call — needed for the English-scan vision fallback (see documents/pipeline.py)."""

    def __init__(self, api_key: str, model: str, timeout: float, name: str = "gemini"):
        # `name` is settable (not just the "gemini" class default) so that a
        # second/third Gemini key registered under "gemini_2"/"gemini_3" (see
        # llm/factory.py) gets its own identity in llm/chain.py's per-provider
        # cooldown map — otherwise two GeminiProvider instances would both
        # report as "gemini" and silently share one cooldown entry, so a 429
        # on key 1 would incorrectly cool down key 2 as well, defeating the
        # entire point of having a second key.
        self.name = name
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system: str,
        user_text: str,
        image: bytes | None = None,
        image_media_type: str | None = None,
        want_json: bool = False,
    ) -> str:
        parts: list[dict] = [{"text": user_text}]
        if image is not None:
            parts.append(
                {"inline_data": {"mime_type": image_media_type or "image/png", "data": base64.b64encode(image).decode("ascii")}}
            )
        generation_config = {"temperature": 0.2}
        if want_json:
            generation_config["responseMimeType"] = "application/json"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        try:
            # The key goes in a header, not the `?key=` query param the REST
            # docs lead with — a query param lands in request URLs that show
            # up verbatim in httpx's own exception messages (and in most HTTP
            # client/proxy logs), so this keeps the key out of that surface
            # entirely rather than relying only on redacting it after the fact.
            resp = httpx.post(
                url, headers={"x-goog-api-key": self.api_key}, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise classify_http_error("gemini", exc, self.api_key) from exc

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LlmProviderError(f"gemini returned an unexpected response shape: {data}") from exc


class OpenAiCompatibleProvider(LlmProvider):
    """Groq and the Hugging Face router both speak the OpenAI chat-completions
    shape, so one implementation covers both — only the base URL, auth, and
    per-provider quirks (vision support, JSON mode support) differ."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        supports_vision: bool = False,
        supports_json_mode: bool = True,
    ):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.supports_vision = supports_vision
        self.supports_json_mode = supports_json_mode

    def generate(
        self,
        system: str,
        user_text: str,
        image: bytes | None = None,
        image_media_type: str | None = None,
        want_json: bool = False,
    ) -> str:
        if image is not None and not self.supports_vision:
            raise LlmProviderError(f"{self.name} model {self.model!r} isn't configured for image input")

        if image is not None:
            user_content: object = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type or 'image/png'};base64,{base64.b64encode(image).decode('ascii')}"
                    },
                },
            ]
        else:
            user_content = user_text

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
            "temperature": 0.2,
        }
        if want_json and self.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise classify_http_error(self.name, exc, self.api_key) from exc

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LlmProviderError(f"{self.name} returned an unexpected response shape: {data}") from exc


class OllamaProvider(LlmProvider):
    """Local model served by Ollama (https://ollama.com) — no API key, no
    rate limit, no hard request-size cap the way Groq/HF's gateways have.
    Text-only here (small local models like Qwen aren't wired for image
    input in this app); bounded by this machine's RAM and the model's context
    window rather than anything network-side, which is why `num_ctx` is set
    explicitly — Ollama's own default is far smaller than a document-
    extraction chunk needs.

    `num_thread=0` means "use every logical core" (resolved at call time via
    os.cpu_count(), not baked in at construction, in case the process ever
    moves to different hardware). `num_predict` is a generous output-length
    ceiling — see config.py's ollama_num_predict for why it's a safety net,
    not a normal-case constraint: the real lever for a slow local model is
    keeping the PROMPT asking for a short answer in the first place (see
    documents/extractor.py's extract_section_fields), not capping tokens
    after the fact and risking a truncated real answer."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        num_ctx: int = 8192,
        num_thread: int = 0,
        num_predict: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_thread = num_thread
        self.num_predict = num_predict

    def generate(
        self,
        system: str,
        user_text: str,
        image: bytes | None = None,
        image_media_type: str | None = None,
        want_json: bool = False,
    ) -> str:
        if image is not None:
            raise LlmProviderError(f"ollama model {self.model!r} isn't configured for image input")

        options = {
            "num_ctx": self.num_ctx,
            "num_thread": self.num_thread or (os.cpu_count() or 4),
            "num_predict": self.num_predict,
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            "stream": False,
            "options": options,
        }
        if want_json:
            payload["format"] = "json"

        try:
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise classify_http_error("ollama", exc) from exc

        data = resp.json()
        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise LlmProviderError(f"ollama returned an unexpected response shape: {data}") from exc
