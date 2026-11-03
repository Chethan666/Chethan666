"""Model clients: Anthropic, OpenAI, Google (Gemini), xAI, and Amazon Bedrock.

Every client returns validated JSON for a task. SDKs are imported lazily so the
pipeline and its tests load without them.

Amazon Bedrock ("aws:" specs) serves several companies' models through one
Bedrock API key: Claude through Bedrock's Anthropic Messages endpoint, and
OpenAI and xAI models through its OpenAI-compatible endpoint. Bedrock has no
Claude web search, so research stays on the direct Claude API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .llm import LLMError, ModelSpec, UsageRecorder, parse_json

# Claude models that take the server-side refusal fallback.
FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")


class AnthropicClient:
    """Claude via the official SDK, streamed because drafts can run long."""

    supports_web_search = True

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        import anthropic

        self.anthropic = anthropic
        self.spec = spec
        self.recorder = recorder
        self.max_tokens = cfg["providers"]["anthropic"]["max_tokens"]
        self.client = anthropic.Anthropic(timeout=cfg["providers"]["anthropic"]["timeout_s"], max_retries=4)

    def _request(self, task: str, **kwargs: Any) -> Any:
        if self.spec.model in FALLBACK_MODELS:
            # Re-runs a declined request on Anthropic's recommended fallback model.
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if self.spec.effort:
            kwargs.setdefault("output_config", {})["effort"] = self.spec.effort
        try:
            with self.client.beta.messages.stream(model=self.spec.model, max_tokens=self.max_tokens,
                                                  **kwargs) as stream:
                message = stream.get_final_message()
        except self.anthropic.APIStatusError as exc:
            raise LLMError(f"{self.spec.label} HTTP {exc.status_code}: {exc.message}") from exc
        except self.anthropic.APIConnectionError as exc:
            raise LLMError(f"{self.spec.label} connection failed: {exc}") from exc
        self.recorder.record(task, self.spec,
                             message.usage.input_tokens, message.usage.output_tokens)
        if message.stop_reason == "refusal":
            raise LLMError(f"{self.spec.label} declined the request")
        if message.stop_reason == "max_tokens":
            raise LLMError(f"{self.spec.label} hit max_tokens before finishing")
        return message

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.recorder.check()
        message = self._request(
            task,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in message.content if b.type == "text"), "")
        return parse_json(text)

    def research(self, task: str, system: str, prompt: str, max_searches: int = 8) -> str:
        """Free-text research with Claude's server-side web search."""
        self.recorder.check()
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}]
        for _ in range(6):
            message = self._request(task, system=system, messages=messages, tools=tools)
            if message.stop_reason != "pause_turn":
                return "\n".join(b.text for b in message.content if b.type == "text").strip()
            # A long search turn paused; send it back so the server continues it.
            messages.append({"role": "assistant", "content": message.content})
        raise LLMError(f"{self.spec.label} research did not finish")

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list()]


class OpenAIClient:
    """OpenAI via the official SDK's Responses API with strict JSON schemas."""

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        import openai

        self.openai = openai
        self.spec = spec
        self.recorder = recorder
        self.max_tokens = cfg["providers"]["openai"]["max_tokens"]
        self.dimensions = cfg["providers"]["openai"].get("embedding_dimensions")
        self.client = openai.OpenAI(timeout=cfg["providers"]["openai"]["timeout_s"], max_retries=4)

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.recorder.check()
        kwargs: dict[str, Any] = dict(
            model=self.spec.model, instructions=system, input=prompt, max_output_tokens=self.max_tokens,
            text={"format": {"type": "json_schema", "name": task, "schema": schema, "strict": True}},
        )
        if self.spec.effort:
            kwargs["reasoning"] = {"effort": self.spec.effort}
        try:
            response = self.client.responses.create(**kwargs)
        except self.openai.APIStatusError as exc:
            raise LLMError(f"{self.spec.label} HTTP {exc.status_code}: {exc.message}") from exc
        except self.openai.APIConnectionError as exc:
            raise LLMError(f"{self.spec.label} connection failed: {exc}") from exc
        if response.usage:
            self.recorder.record(task, self.spec, response.usage.input_tokens, response.usage.output_tokens)
        if response.status != "completed":
            raise LLMError(f"{self.spec.label} response {response.status}")
        return parse_json(response.output_text)

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {"model": self.spec.model, "input": text}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        result = self.client.embeddings.create(**kwargs)
        self.recorder.record("embed", self.spec, result.usage.prompt_tokens, 0)
        return list(result.data[0].embedding)

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())


class GoogleClient:
    """Gemini over its REST API (text JSON and embeddings)."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        self.spec = spec
        self.recorder = recorder
        self.cfg = cfg["providers"]["google"]
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("Set GEMINI_API_KEY in the environment")

    def _post(self, path: str, body: dict | None, method: str = "POST") -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(5):
            request = urllib.request.Request(self.BASE_URL + path, data=data, method=method, headers={
                "Content-Type": "application/json", "x-goog-api-key": self.api_key})
            try:
                with urllib.request.urlopen(request, timeout=self.cfg["timeout_s"]) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if (exc.code != 429 and exc.code < 500) or attempt == 4:
                    raise LLMError(f"{self.spec.label} HTTP {exc.code}: "
                                   f"{exc.read().decode(errors='replace')[:300]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 4:
                    raise LLMError(f"{self.spec.label} request failed: {exc}") from exc
            time.sleep(2 ** (attempt + 2))
        raise AssertionError("unreachable")

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.recorder.check()
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema},
        }
        try:
            data = self._post(f"/models/{self.spec.model}:generateContent", body)
        except LLMError as exc:
            if "HTTP 400" not in str(exc):
                raise
            # Older models reject some schema keywords; ask for the shape in words instead.
            del body["generationConfig"]["responseJsonSchema"]
            body["contents"][0]["parts"][0]["text"] = (
                f"{prompt}\n\nReturn JSON matching this JSON Schema exactly:\n{json.dumps(schema)}")
            data = self._post(f"/models/{self.spec.model}:generateContent", body)
        usage = data.get("usageMetadata", {})
        self.recorder.record(task, self.spec, int(usage.get("promptTokenCount", 0)),
                             int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0)))
        candidates = data.get("candidates") or []
        if not candidates or not candidates[0].get("content", {}).get("parts"):
            reason = candidates[0].get("finishReason") if candidates else data.get("promptFeedback")
            raise LLMError(f"{self.spec.label} returned no text ({reason})")
        return parse_json("".join(p.get("text", "") for p in candidates[0]["content"]["parts"]))

    def embed(self, text: str) -> list[float]:
        body = {"content": {"parts": [{"text": text}]}, "taskType": "SEMANTIC_SIMILARITY",
                "outputDimensionality": self.cfg["embedding_dimensions"]}
        data = self._post(f"/models/{self.spec.model}:embedContent", body)
        self.recorder.record("embed", self.spec, len(text.split()), 0)
        return [float(v) for v in data["embedding"]["values"]]

    def list_models(self) -> list[str]:
        data = self._post("/models?pageSize=1000", None, method="GET")
        return sorted(m["name"].removeprefix("models/") for m in data.get("models", []))


class XAIClient:
    """xAI Grok through its OpenAI-compatible Chat Completions endpoint.

    Uses strict JSON schemas; if the model rejects them, retries once asking for
    JSON in words (the same fallback the Gemini client uses).
    """

    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        import openai

        self.openai = openai
        self.spec = spec
        self.recorder = recorder
        settings = cfg["providers"]["xai"]
        self.max_tokens = settings["max_tokens"]
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise LLMError("Set XAI_API_KEY in the environment")
        self.client = openai.OpenAI(api_key=api_key, base_url=self.BASE_URL,
                                    timeout=settings["timeout_s"], max_retries=4)

    def _complete(self, task: str, messages: list[dict], response_format: dict) -> Any:
        try:
            response = self.client.chat.completions.create(
                model=self.spec.model, messages=messages, response_format=response_format,
                max_completion_tokens=self.max_tokens)
        except self.openai.APIConnectionError as exc:
            raise LLMError(f"{self.spec.label} connection failed: {exc}") from exc
        if response.usage:
            self.recorder.record(task, self.spec, response.usage.prompt_tokens, response.usage.completion_tokens)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"{self.spec.label} hit max tokens before finishing")
        return choice.message.content or ""

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.recorder.check()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        try:
            text = self._complete(task, messages, {"type": "json_schema", "json_schema": {
                "name": task, "schema": schema, "strict": True}})
        except self.openai.BadRequestError:
            messages[1]["content"] = f"{prompt}\n\nReturn JSON matching this JSON Schema exactly:\n{json.dumps(schema)}"
            try:
                text = self._complete(task, messages, {"type": "json_object"})
            except self.openai.APIStatusError as exc:
                raise LLMError(f"{self.spec.label} HTTP {exc.status_code}: {exc.message}") from exc
        except self.openai.APIStatusError as exc:
            raise LLMError(f"{self.spec.label} HTTP {exc.status_code}: {exc.message}") from exc
        return parse_json(text)

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())


def _aws_region(cfg: dict) -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or cfg["providers"]["aws"]["region"]


def _aws_key() -> str:
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not key:
        raise LLMError("Set AWS_BEARER_TOKEN_BEDROCK (an Amazon Bedrock API key) in the environment")
    return key


def _aws_openai_urls(cfg: dict) -> list[str]:
    """The configured OpenAI-compatible Bedrock address first, then the alternative.

    AWS documents both /openai/v1 and /v1 for this endpoint, so a 404 on one
    is retried on the other.
    """
    region = _aws_region(cfg)
    configured = cfg["providers"]["aws"]["openai_base_url"].format(region=region)
    host = f"https://bedrock-mantle.{region}.api.aws"
    return [configured] + [host + path for path in ("/openai/v1", "/v1") if host + path != configured]


def _aws_openai_url(cfg: dict) -> str:
    return _aws_openai_urls(cfg)[0]


class _BedrockAddressFallback:
    """Retries a 404 once on Bedrock's alternative OpenAI-compatible address, then keeps it."""

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            return super().generate_json(task, system, prompt, schema)
        except LLMError as exc:
            alternatives = [u for u in _aws_openai_urls(self.cfg) if u.rstrip("/") != str(self.client.base_url).rstrip("/")]
            if "HTTP 404" not in str(exc) or not alternatives or getattr(self, "_switched", False):
                raise
            self._switched = True
            self.client = self.client.with_options(base_url=alternatives[0])
            return super().generate_json(task, system, prompt, schema)


class AWSClaudeClient(AnthropicClient):
    """Claude on Amazon Bedrock via the SDK's Bedrock Mantle client (same Messages API)."""

    supports_web_search = False  # Bedrock doesn't offer Claude's web search tool

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        import anthropic

        self.anthropic = anthropic
        self.spec = spec
        self.recorder = recorder
        settings = cfg["providers"]["aws"]
        self.max_tokens = settings["max_tokens"]
        self.cfg = cfg
        self.client = anthropic.AnthropicBedrockMantle(api_key=_aws_key(), aws_region=_aws_region(cfg),
                                                       timeout=settings["timeout_s"], max_retries=4)

    def research(self, task: str, system: str, prompt: str, max_searches: int = 8) -> str:
        raise LLMError("Claude web search isn't available on Amazon Bedrock; use an anthropic: researcher")

    def list_models(self) -> list[str]:
        return aws_models(self.cfg)


class AWSResponsesClient(_BedrockAddressFallback, OpenAIClient):
    """OpenAI models on Amazon Bedrock through its OpenAI-compatible Responses API."""

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        import openai

        self.openai = openai
        self.spec = spec
        self.recorder = recorder
        settings = cfg["providers"]["aws"]
        self.max_tokens = settings["max_tokens"]
        self.dimensions = None
        self.cfg = cfg
        self.client = openai.OpenAI(api_key=_aws_key(), base_url=_aws_openai_url(cfg),
                                    timeout=settings["timeout_s"], max_retries=4)

    def list_models(self) -> list[str]:
        return aws_models(self.cfg)


class AWSChatClient(_BedrockAddressFallback, XAIClient):
    """Other companies' models on Amazon Bedrock (such as xAI Grok) through Chat Completions."""

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        import openai

        self.openai = openai
        self.spec = spec
        self.recorder = recorder
        settings = cfg["providers"]["aws"]
        self.max_tokens = settings["max_tokens"]
        self.cfg = cfg
        self.client = openai.OpenAI(api_key=_aws_key(), base_url=_aws_openai_url(cfg),
                                    timeout=settings["timeout_s"], max_retries=4)

    def list_models(self) -> list[str]:
        return aws_models(self.cfg)


class AWSEmbedClient:
    """Amazon Titan text embeddings on Bedrock (InvokeModel), authenticated with the Bedrock API key."""

    def __init__(self, spec: ModelSpec, cfg: dict, recorder: UsageRecorder):
        self.spec = spec
        self.recorder = recorder
        self.cfg = cfg
        self.key = _aws_key()
        self.url = (f"https://bedrock-runtime.{_aws_region(cfg)}.amazonaws.com/model/"
                    f"{urllib.parse.quote(spec.model, safe='')}/invoke")

    def embed(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True}).encode()
        for attempt in range(5):
            request = urllib.request.Request(self.url, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                "Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                if (exc.code != 429 and exc.code < 500) or attempt == 4:
                    raise LLMError(f"{self.spec.label} HTTP {exc.code}: "
                                   f"{exc.read().decode(errors='replace')[:300]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 4:
                    raise LLMError(f"{self.spec.label} request failed: {exc}") from exc
            time.sleep(2 ** (attempt + 2))
        self.recorder.record("embed", self.spec, int(data.get("inputTextTokenCount", 0)), 0)
        return [float(v) for v in data["embedding"]]

    def list_models(self) -> list[str]:
        return aws_models(self.cfg)


def aws_models(cfg: dict) -> list[str]:
    """Model IDs listed by Bedrock's OpenAI-compatible endpoint (whichever address answers)."""
    import openai

    last: Exception | None = None
    for url in _aws_openai_urls(cfg):
        try:
            client = openai.OpenAI(api_key=_aws_key(), base_url=url, timeout=60)
            return sorted(m.id for m in client.models.list())
        except openai.NotFoundError as exc:
            last = exc
    raise LLMError(f"Bedrock model list not found at {', '.join(_aws_openai_urls(cfg))}: {last}")


PROVIDERS = {"anthropic": AnthropicClient, "openai": OpenAIClient, "google": GoogleClient, "xai": XAIClient}


def build(spec: ModelSpec, cfg: dict, recorder: UsageRecorder) -> Any:
    if spec.provider == "aws":
        if spec.family == "anthropic":
            return AWSClaudeClient(spec, cfg, recorder)
        if spec.family == "amazon" and "embed" in spec.model:
            return AWSEmbedClient(spec, cfg, recorder)
        if any(spec.model.startswith(p) for p in cfg["providers"]["aws"]["responses_api_models"]):
            return AWSResponsesClient(spec, cfg, recorder)
        return AWSChatClient(spec, cfg, recorder)
    return PROVIDERS[spec.provider](spec, cfg, recorder)
