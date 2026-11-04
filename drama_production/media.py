"""Paid media services: fal.ai for pictures and video, Boson AI for voices.

Both are plain HTTPS calls (no SDK). Keys come only from the environment (FAL_KEY,
BOSON_API_KEY). Model names live in the config, because these catalogs change often;
`python -m drama_production check` lists what each configured model expects before anything is spent.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class MediaError(RuntimeError):
    pass


def _request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, body: Any = None,
             timeout: int = 120, raw: bool = False) -> Any:
    data = None
    headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise MediaError(f"{method} {url.split('?')[0]}: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise MediaError(f"{method} {url.split('?')[0]} failed: {exc}") from exc
    return payload if raw else json.loads(payload or b"{}")


def data_uri(path: Path) -> str:
    kind = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_request(url, timeout=600, raw=True))
    return path


class Fal:
    """fal.ai's queue API: submit, poll the status URL, fetch the result."""

    QUEUE = "https://queue.fal.run"

    def __init__(self, key: str | None = None, poll_seconds: float = 3.0, timeout_seconds: int = 1200):
        self.key = key or os.environ.get("FAL_KEY", "")
        if not self.key:
            raise MediaError("Set FAL_KEY (a fal.ai API key) in the environment")
        self.poll = poll_seconds
        self.timeout = timeout_seconds

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.key}", "Accept": "application/json"}

    def run(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
        submitted = _request(f"{self.QUEUE}/{endpoint}", method="POST", headers=self.headers, body=arguments)
        status_url, response_url = submitted.get("status_url"), submitted.get("response_url")
        if not status_url or not response_url:
            raise MediaError(f"{endpoint}: unexpected submit reply {str(submitted)[:300]}")
        deadline = time.time() + self.timeout
        while True:
            status = _request(status_url, headers=self.headers)
            state = status.get("status")
            if state == "COMPLETED":
                break
            if state not in ("IN_QUEUE", "IN_PROGRESS"):
                raise MediaError(f"{endpoint}: {state}: {str(status)[:300]}")
            if time.time() > deadline:
                raise MediaError(f"{endpoint}: no result after {self.timeout} s")
            time.sleep(self.poll)
        result = _request(response_url, headers=self.headers)
        if isinstance(result, dict) and result.get("detail") and not (result.get("images") or result.get("video")):
            raise MediaError(f"{endpoint}: {str(result['detail'])[:300]}")  # a failed job also says COMPLETED
        return result

    def schema(self, endpoint: str) -> dict[str, Any]:
        """The model's input fields, from fal's public OpenAPI description."""
        url = "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=" + urllib.parse.quote(endpoint, safe="")
        spec = _request(url)
        schemas = spec.get("components", {}).get("schemas", {})
        inputs = next((v for k, v in schemas.items() if k.endswith("Input")), {})
        return {"fields": sorted(inputs.get("properties", {})), "required": inputs.get("required", [])}


def first_url(result: dict[str, Any], kind: str) -> str:
    """The first image or video URL in a fal result, whatever the model calls it."""
    if kind == "image":
        images = result.get("images") or ([result["image"]] if result.get("image") else [])
        if images and isinstance(images[0], dict) and images[0].get("url"):
            return images[0]["url"]
    else:
        video = result.get("video")
        if isinstance(video, dict) and video.get("url"):
            return video["url"]
        if isinstance(video, str):
            return video
    raise MediaError(f"no {kind} in result: {str(result)[:300]}")


class BosonVoice:
    """Boson AI's Higgs TTS: expressive speech, one fixed voice per character."""

    URL = "https://api.boson.ai/v1/audio"

    def __init__(self, model: str = "higgs-tts-3", key: str | None = None):
        self.key = key or os.environ.get("BOSON_API_KEY", "")
        if not self.key:
            raise MediaError("Set BOSON_API_KEY in the environment")
        self.model = model

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    def speak(self, text: str, voice: str | None, path: Path) -> Path:
        body: dict[str, Any] = {"model": self.model, "input": text, "response_format": "wav"}
        if voice:
            body["voice"] = voice
        path.write_bytes(_request(f"{self.URL}/speech", method="POST", headers=self.headers, body=body,
                                  timeout=300, raw=True))
        return path

    def voices(self) -> Any:
        return _request(f"{self.URL}/voices", headers=self.headers)
