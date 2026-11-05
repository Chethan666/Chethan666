"""Publishing to YouTube, only after a person has approved the video, and reading back how it did.

Uses the YouTube Data API with an OAuth refresh token for the channel (GitHub secrets
YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN). Uploads are private, or
scheduled with `publish_at`, never public straight away: the last look is on YouTube itself.
The title, description with chapters, tags, captions and the first thumbnail go up together;
the other thumbnails are kept for YouTube's Test & Compare, which is set in YouTube Studio.
Stats come back daily into the performance table, which the story factory learns from.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/youtube/v3"
UPLOAD = "https://www.googleapis.com/upload/youtube/v3"
ANALYTICS = "https://youtubeanalytics.googleapis.com/v2/reports"
CATEGORY_ENTERTAINMENT = "24"


class YouTubeError(RuntimeError):
    pass


def _http(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, data: bytes | None = None,
          timeout: int = 600) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        raise YouTubeError(f"{method} {url.split('?')[0]}: HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise YouTubeError(f"{method} {url.split('?')[0]} failed: {exc}") from exc


def configured() -> bool:
    return all(os.environ.get(k) for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"))


class YouTube:
    def __init__(self, http=_http):
        if not configured():
            raise YouTubeError("Set YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET and YOUTUBE_REFRESH_TOKEN (GitHub secrets)")
        self.http = http
        body = urllib.parse.urlencode({
            "client_id": os.environ["YOUTUBE_CLIENT_ID"], "client_secret": os.environ["YOUTUBE_CLIENT_SECRET"],
            "refresh_token": os.environ["YOUTUBE_REFRESH_TOKEN"], "grant_type": "refresh_token"}).encode()
        _, _, payload = self.http(TOKEN_URL, method="POST", data=body,
                                  headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.token = json.loads(payload).get("access_token", "")
        if not self.token:
            raise YouTubeError("no access token in the OAuth reply")

    @property
    def auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def upload(self, video: Path, meta: dict[str, Any], privacy: str = "private", publish_at: str | None = None) -> str:
        status: dict[str, Any] = {"privacyStatus": "private" if publish_at else privacy,
                                  "selfDeclaredMadeForKids": bool(meta.get("made_for_kids", False)),
                                  "containsSyntheticMedia": bool(meta.get("synthetic_media", False))}
        if publish_at:
            status["publishAt"] = publish_at
        description = str(meta.get("description", ""))
        if meta.get("credit"):
            description += f"\n\n{meta['credit']}"
        body = {"snippet": {"title": str(meta.get("title", ""))[:100], "description": description[:5000],
                            "tags": [str(t) for t in meta.get("tags", [])][:30], "categoryId": CATEGORY_ENTERTAINMENT},
                "status": status}
        size = video.stat().st_size
        _, headers, _ = self.http(f"{UPLOAD}/videos?uploadType=resumable&part=snippet,status", method="POST",
                                  data=json.dumps(body).encode(),
                                  headers={**self.auth, "Content-Type": "application/json; charset=UTF-8",
                                           "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)})
        location = next((v for k, v in headers.items() if k.lower() == "location"), None)
        if not location:
            raise YouTubeError("YouTube gave no upload address")
        _, _, payload = self.http(location, method="PUT", data=video.read_bytes(),
                                  headers={**self.auth, "Content-Type": "video/mp4"}, timeout=3600)
        video_id = json.loads(payload).get("id")
        if not video_id:
            raise YouTubeError(f"upload finished without a video id: {payload[:200]!r}")
        return video_id

    def thumbnail(self, video_id: str, image: Path) -> None:
        kind = "image/jpeg" if image.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        self.http(f"{UPLOAD}/thumbnails/set?videoId={video_id}", method="POST", data=image.read_bytes(),
                  headers={**self.auth, "Content-Type": kind})

    def captions(self, video_id: str, srt: Path, language: str = "en", name: str = "English") -> None:
        boundary = "hollowlinesboundary"
        meta = json.dumps({"snippet": {"videoId": video_id, "language": language, "name": name, "isDraft": False}})
        data = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
                f"--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n").encode() + srt.read_bytes() + \
            f"\r\n--{boundary}--\r\n".encode()
        self.http(f"{UPLOAD}/captions?uploadType=multipart&part=snippet", method="POST", data=data,
                  headers={**self.auth, "Content-Type": f"multipart/related; boundary={boundary}"})

    def comment(self, video_id: str, text: str) -> None:
        body = {"snippet": {"videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": text}}}}
        self.http(f"{API}/commentThreads?part=snippet", method="POST", data=json.dumps(body).encode(),
                  headers={**self.auth, "Content-Type": "application/json"})

    def stats(self, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(video_ids), 50):
            ids = ",".join(video_ids[start:start + 50])
            _, _, payload = self.http(f"{API}/videos?part=statistics&id={ids}", headers=self.auth)
            for item in json.loads(payload).get("items", []):
                s = item.get("statistics", {})
                out[item["id"]] = {"views": int(s.get("viewCount", 0)), "likes": int(s.get("likeCount", 0)),
                                   "comments": int(s.get("commentCount", 0))}
        for vid in out:  # retention needs the Analytics API (the yt-analytics.readonly scope)
            try:
                query = urllib.parse.urlencode({"ids": "channel==MINE", "startDate": "2020-01-01",
                                                "endDate": "2100-01-01", "metrics": "averageViewPercentage",
                                                "filters": f"video=={vid}"})
                _, _, payload = self.http(f"{ANALYTICS}?{query}", headers=self.auth)
                rows = json.loads(payload).get("rows") or []
                if rows:
                    out[vid]["avg_view_percentage"] = float(rows[0][0])
            except YouTubeError as exc:
                log.info("No retention for %s (%s)", vid, str(exc)[:120])
        return out


def publish(yt: YouTube, folder: Path, privacy: str = "private", publish_at: str | None = None) -> dict[str, Any]:
    """Upload a finished video folder (video.mp4, metadata.json, captions.srt, thumbnail_1.jpg)."""
    meta = json.loads((folder / "metadata.json").read_text())
    video_id = yt.upload(folder / "video.mp4", meta, privacy, publish_at)
    done = {"video_id": video_id, "privacy": "scheduled" if publish_at else privacy, "steps": []}
    thumb = next((p for p in (folder / "thumbnail_1.jpg", folder / "thumbnail.png") if p.exists()), None)
    for step, call in (("thumbnail", lambda: thumb and yt.thumbnail(video_id, thumb)),
                       ("captions", lambda: (folder / "captions.srt").exists() and
                        yt.captions(video_id, folder / "captions.srt")),
                       # The API can post the comment; pinning it is one click in YouTube Studio.
                       ("comment", lambda: meta.get("pinned_comment") and yt.comment(video_id, meta["pinned_comment"]))):
        try:
            call()
            done["steps"].append(step)
        except YouTubeError as exc:  # the video is up; a missing extra is reported, not fatal
            log.warning("%s failed: %s", step, exc)
            done.setdefault("failed", []).append(f"{step}: {str(exc)[:200]}")
    return done
