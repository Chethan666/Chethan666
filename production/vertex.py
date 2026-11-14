"""Google Cloud Vertex AI: Nano Banana 2.1 pictures and Gemini checks, paid from Google Cloud.

Plain HTTPS calls to Vertex AI's generateContent, signed in with a Google Cloud API key
(GitHub secret GEMINI_PLATFORM_API_KEY) or a service account key (GCP_SERVICE_ACCOUNT_JSON: the
key file's JSON), kept only in the environment. The location is `global` unless GCP_LOCATION
says otherwise; GCP_PROJECT names the project when the key alone does not.

Models are named in the config as "vertex:<model id>" (e.g. "vertex:gemini-nano-banana-2.1").
`use_vertex()` moves the configured roles from fal to Vertex when a key is present; the
`Providers` router sends "vertex:" models to Google and every other model to fal, and if a
Vertex call fails for any reason but a refused picture, the same request goes to the fal
model it replaced. Nothing changes for a run without the key.

Requests and answers follow Vertex AI's generateContent:
  request  {"contents": [{"role": "user", "parts": [{"inlineData": {...}}, ..., {"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"],
                                 "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}}}
  answer   candidates[0].content.parts[].inlineData {"mimeType", "data" (base64)}
A setting the model rejects (HTTP 400 naming imageSize, imageConfig or responseModalities)
is dropped and the request sent again, so a model that takes fewer settings still answers.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import threading
import time
import urllib.parse
from typing import Any, Callable

from production.media import MediaError, _request

log = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
PREFIX = "vertex:"
# Horror is the channel's genre: block only what is clearly harmful, so a dark scene is not refused.
SAFETY = [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT")]


KEY_ENV = "GEMINI_PLATFORM_API_KEY"  # a Google Cloud API key for Vertex AI (not an AI Studio key)


def available() -> bool:
    return bool(os.environ.get(KEY_ENV) or os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
                or os.environ.get("GCP_ACCESS_TOKEN"))


def client() -> "Vertex":
    """Vertex AI signed in with the Google Cloud API key (GEMINI_PLATFORM_API_KEY) when it is set,
    else with a service account key: the same models, requests and answers, billed to the Google
    Cloud project the key belongs to."""
    if os.environ.get(KEY_ENV):
        return VertexKey()
    return Vertex()


def _data(uri: str) -> tuple[str, str]:
    """(mime type, base64 data) of a data: URI."""
    if not uri.startswith("data:") or ";base64," not in uri:
        raise MediaError("vertex: reference pictures must be sent as data URIs")
    head, data = uri[5:].split(";base64,", 1)
    return head or "image/png", data


class Vertex:
    """generateContent on Vertex AI, with a service account's access token."""

    def __init__(self, credentials: dict[str, Any] | None = None, project: str | None = None,
                 location: str | None = None, http: Callable[..., Any] | None = None,
                 token: str | None = None):
        raw = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "")
        self.info = credentials if credentials is not None else (json.loads(raw) if raw.strip() else {})
        self.project = project or os.environ.get("GCP_PROJECT") or self.info.get("project_id", "")
        self.location = location or os.environ.get("GCP_LOCATION") or "global"
        self.http = http or _request
        self._token = token or os.environ.get("GCP_ACCESS_TOKEN") or ""
        self._expiry = float("inf") if self._token else 0.0
        self.lock = threading.Lock()
        if not self.project:
            raise MediaError("Vertex AI: no project (set GCP_PROJECT, or use a key file that names one)")
        if not self._token and not self.info.get("private_key"):
            raise MediaError("Vertex AI: set GCP_SERVICE_ACCOUNT_JSON (a service account key) as a secret")

    @property
    def host(self) -> str:
        return "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"

    def token(self) -> str:
        """An access token from the service account key (a signed JWT exchanged at Google), kept
        until five minutes before it expires."""
        with self.lock:
            if self._token and time.time() < self._expiry - 300:
                return self._token
            from google.auth import crypt, jwt  # google-auth: only needed when a key signs in

            now = int(time.time())
            signer = crypt.RSASigner.from_service_account_info(self.info)
            assertion = jwt.encode(signer, {"iss": self.info["client_email"], "scope": SCOPE, "aud": TOKEN_URL,
                                            "iat": now, "exp": now + 3600})
            body = urllib.parse.urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                           "assertion": assertion.decode() if isinstance(assertion, bytes)
                                           else assertion}).encode()
            import urllib.request
            request = urllib.request.Request(TOKEN_URL, data=body, method="POST",
                                             headers={"Content-Type": "application/x-www-form-urlencoded"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    got = json.loads(response.read())
            except Exception as exc:  # noqa: BLE001 - any sign-in failure is reported the same way
                raise MediaError(f"Vertex AI sign-in failed: {str(exc)[:300]}") from exc
            self._token = got["access_token"]
            self._expiry = now + float(got.get("expires_in", 3600))
            return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"}

    def url(self, model: str, method: str = "generateContent") -> str:
        return (f"https://{self.host}/v1/projects/{self.project}/locations/{self.location}"
                f"/publishers/google/models/{model}:{method}")

    WAITS = (10.0, 30.0, 60.0)  # a busy model or a per-minute quota: wait and ask again before falling back

    def _post(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        for k in range(len(self.WAITS) + 1):
            try:
                return self.http(self.url(model), method="POST", body=body, timeout=300, headers=self._headers())
            except MediaError as exc:
                text = str(exc)
                busy = any(w in text for w in ("HTTP 429", "HTTP 503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if not busy or k == len(self.WAITS):
                    raise
                log.info("Vertex %s busy (%s); asking again in %.0f s", model, text[:120], self.WAITS[k])
                time.sleep(self.WAITS[k])
        raise MediaError(f"vertex {model}: no answer")

    @staticmethod
    def _parts(args: dict[str, Any]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for uri in args.get("image_urls") or []:
            mime, data = _data(str(uri))
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        parts.append({"text": str(args.get("prompt", ""))})
        return parts

    def run(self, endpoint: str, args: dict[str, Any]) -> dict[str, Any]:
        """The same answer shape as fal: {"images": [{"url": data URI}]} for a picture model,
        {"output": text} for a text model."""
        model = endpoint[len(PREFIX):] if endpoint.startswith(PREFIX) else endpoint
        if "image" in model or "banana" in model:
            return self.image(model, args)
        return self.text(model, args)

    def image(self, model: str, args: dict[str, Any]) -> dict[str, Any]:
        image_config: dict[str, Any] = {}
        if args.get("aspect_ratio"):
            image_config["aspectRatio"] = str(args["aspect_ratio"])
        if args.get("resolution"):
            image_config["imageSize"] = str(args["resolution"])
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": self._parts(args)}],
                                "generationConfig": {"responseModalities": ["IMAGE"]},
                                "safetySettings": SAFETY}
        if image_config:
            body["generationConfig"]["imageConfig"] = image_config
        if args.get("system_prompt"):
            body["systemInstruction"] = {"parts": [{"text": str(args["system_prompt"])}]}
        tries = [body]
        if "imageSize" in image_config:  # a model that takes no size still answers at its own
            smaller = copy.deepcopy(body)
            smaller["generationConfig"]["imageConfig"].pop("imageSize")
            tries.append(smaller)
        both = copy.deepcopy(tries[-1])
        both["generationConfig"]["responseModalities"] = ["TEXT", "IMAGE"]
        tries.append(both)
        last: MediaError | None = None
        for k, attempt in enumerate(tries):
            try:
                reply = self._post(model, attempt)
            except MediaError as exc:
                text = str(exc)
                if "HTTP 400" in text and any(w in text for w in ("imageSize", "image_size", "imageConfig",
                                                                 "responseModalities", "response_modalities")):
                    last = exc
                    log.info("Vertex %s rejected a setting, asking again without it: %s", model, text[:200])
                    continue
                raise
            for candidate in reply.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and inline.get("data"):
                        mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                        return {"images": [{"url": f"data:{mime};base64,{inline['data']}"}],
                                "usage": reply.get("usageMetadata", {})}
            finish = ", ".join(str(c.get("finishReason", "")) for c in reply.get("candidates") or [])
            blocked = (reply.get("promptFeedback") or {}).get("blockReason", "")
            # "content" in the message: the studio does not ask again for a refused picture
            raise MediaError(f"vertex {model}: no picture, content refused or empty "
                             f"(finish: {finish or 'none'}{', blocked: ' + blocked if blocked else ''})")
        raise last or MediaError(f"vertex {model}: no picture")

    def text(self, model: str, args: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": self._parts(args)}],
                                "generationConfig": {"maxOutputTokens": int(args.get("max_tokens") or 2048),
                                                     "temperature": float(args.get("temperature") or 0.2)},
                                "safetySettings": SAFETY}
        if args.get("system_prompt"):
            body["systemInstruction"] = {"parts": [{"text": str(args["system_prompt"])}]}
        reply = self._post(model, body)
        texts = [p.get("text", "") for c in reply.get("candidates") or []
                 for p in (c.get("content") or {}).get("parts") or [] if p.get("text")]
        if not texts:
            raise MediaError(f"vertex {model}: no answer ({str(reply)[:200]})")
        return {"output": "".join(texts)}

    def model_info(self, model: str) -> dict[str, Any]:
        """The model's record in Vertex's catalogue (free): proves the key, the API and the model."""
        return self.http(f"https://{self.host}/v1/publishers/google/models/{model}", timeout=60,
                         headers={"Authorization": f"Bearer {self.token()}"})


class VertexKey(Vertex):
    """Vertex AI with a Google Cloud API key (sent as the x-goog-api-key header, never in the URL):
    no service account. Without GCP_PROJECT the project-less address is used, which Vertex AI
    resolves from the key; with it, the project's own address."""

    def __init__(self, key: str | None = None, project: str | None = None, location: str | None = None,
                 http: Callable[..., Any] | None = None):
        self.key = key or os.environ.get(KEY_ENV, "")
        if not self.key:
            raise MediaError(f"Vertex AI: set {KEY_ENV} (a Google Cloud API key) as a secret")
        self.project = project or os.environ.get("GCP_PROJECT", "")
        self.location = location or os.environ.get("GCP_LOCATION") or "global"
        self.http = http or _request
        self.lock = threading.Lock()

    def token(self) -> str:
        return ""

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.key, "Content-Type": "application/json"}

    def url(self, model: str, method: str = "generateContent") -> str:
        if self.project:
            return super().url(model, method)
        return f"https://{self.host}/v1/publishers/google/models/{model}:{method}"

    def model_info(self, model: str) -> dict[str, Any]:
        return self.http(f"https://{self.host}/v1/publishers/google/models/{model}", timeout=60,
                         headers={"x-goog-api-key": self.key})


class Providers:
    """fal and Vertex behind one `run()`: "vertex:" models go to Google, every other to fal.
    A Vertex failure (not a refused picture) is sent to the fal model it replaced."""

    def __init__(self, fal: Any = None, vertex: Vertex | None = None,
                 fallback: dict[str, list[str]] | None = None):
        self.fal, self.vertex, self.fallback = fal, vertex, dict(fallback or {})
        self.fell_back = 0

    def run(self, endpoint: str, args: dict[str, Any]) -> dict[str, Any]:
        if not endpoint.startswith(PREFIX):
            if self.fal is None:
                raise MediaError(f"{endpoint}: no FAL_KEY")
            return self.fal.run(endpoint, args)
        try:
            if self.vertex is None:
                raise MediaError("Vertex AI: no key")
            return self.vertex.run(endpoint, args)
        except MediaError as exc:
            spares = self.fallback.get(endpoint) or []
            # with reference pictures, the fal model that takes them (".../edit"); without, the other
            refs = bool(args.get("image_urls"))
            spare = next((e for e in spares if e.endswith("/edit") == refs), spares[0] if spares else "")
            if self.fal is None or not spare or "content refused" in str(exc):
                raise
            self.fell_back += 1
            log.warning("Vertex %s failed (%s); sent to %s on fal", endpoint, str(exc)[:200], spare)
            return self.fal.run(spare, args)

    def __getattr__(self, name: str) -> Any:  # schema(), price() and the rest stay fal's
        if self.fal is None:
            raise AttributeError(name)
        return getattr(self.fal, name)


def use_vertex(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """The stills config with its Nano Banana 2.1 roles (and the checker) moved to Vertex, and
    the fal model each Vertex model replaced. Unchanged when `vertex.enabled` is off."""
    settings = cfg.get("vertex") or {}
    if not settings.get("enabled", True):
        return cfg, {}
    cfg = copy.deepcopy(cfg)
    models: dict[str, str] = cfg.get("models", {})
    fallback: dict[str, list[str]] = {}
    moves = {str(k): str(v) for k, v in (settings.get("models") or {}).items()}  # fal endpoint -> vertex model
    for role, endpoint in list(models.items()):
        target = moves.get(endpoint)
        if target:
            models[role] = PREFIX + target
            if endpoint not in fallback.setdefault(PREFIX + target, []):
                fallback[PREFIX + target].append(endpoint)
    prices = cfg.setdefault("prices", {})
    for target, usd in (settings.get("prices") or {}).items():
        prices[PREFIX + str(target)] = float(usd)
    return cfg, fallback
