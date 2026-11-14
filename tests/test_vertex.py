"""Vertex AI: the request it sends, the answers it reads, the fallback to fal, the config move."""

import base64
import unittest

from production.media import MediaError
from production.vertex import Providers, Vertex, use_vertex

PNG = base64.b64encode(b"png").decode()


def answer_picture():
    return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}]}}]}


class VertexTests(unittest.TestCase):
    def client(self, replies):
        calls = []

        def http(url, method="GET", headers=None, body=None, timeout=0, raw=False):
            calls.append((url, body, headers))
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return Vertex(credentials={}, project="proj", token="tok", http=http), calls

    def test_a_picture_is_asked_with_its_references_size_and_shape(self):
        v, calls = self.client([answer_picture()])
        got = v.run("vertex:gemini-nano-banana-2.1", {"prompt": "a bus", "aspect_ratio": "16:9", "resolution": "2K",
                                                       "image_urls": ["data:image/jpeg;base64,AAAA"]})
        self.assertEqual(got["images"][0]["url"], f"data:image/png;base64,{PNG}")
        url, body, headers = calls[0]
        self.assertEqual(url, "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/publishers/google/"
                              "models/gemini-nano-banana-2.1:generateContent")
        self.assertEqual(headers["Authorization"], "Bearer tok")
        parts = body["contents"][0]["parts"]
        self.assertEqual(parts[0], {"inlineData": {"mimeType": "image/jpeg", "data": "AAAA"}})
        self.assertEqual(parts[-1], {"text": "a bus"})  # the references first, then the words
        self.assertEqual(body["generationConfig"], {"responseModalities": ["IMAGE"],
                                                    "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}})

    def test_a_setting_the_model_rejects_is_dropped_and_the_picture_asked_again(self):
        v, calls = self.client([MediaError("POST x: HTTP 400: imageSize is not supported"), answer_picture()])
        v.run("vertex:gemini-nano-banana-2.1", {"prompt": "x", "aspect_ratio": "16:9", "resolution": "2K"})
        self.assertNotIn("imageSize", calls[1][1]["generationConfig"]["imageConfig"])

    def test_a_refused_picture_says_content_so_it_is_not_paid_for_again(self):
        v, _ = self.client([{"candidates": [{"finishReason": "SAFETY"}]}])
        with self.assertRaises(MediaError) as e:
            v.run("vertex:gemini-nano-banana-2.1", {"prompt": "x"})
        self.assertIn("content refused", str(e.exception))
        self.assertIn("SAFETY", str(e.exception))

    def test_the_checker_reads_gemini_text_with_its_system_prompt(self):
        v, calls = self.client([{"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}])
        got = v.run("vertex:gemini-2.5-flash", {"prompt": "look", "system_prompt": "judge",
                                                 "image_urls": ["data:image/jpeg;base64,AAAA"], "model": "ignored"})
        self.assertEqual(got, {"output": '{"ok": true}'})
        self.assertEqual(calls[0][1]["systemInstruction"], {"parts": [{"text": "judge"}]})

    def test_a_busy_model_is_asked_again_after_a_wait(self):
        v, calls = self.client([MediaError("POST x: HTTP 429: RESOURCE_EXHAUSTED"), answer_picture()])
        v.WAITS = (0.0, 0.0, 0.0)
        v.run("vertex:gemini-nano-banana-2.1", {"prompt": "x"})
        self.assertEqual(len(calls), 2)
        v2, _ = self.client([MediaError("HTTP 429")] * 4)
        v2.WAITS = (0.0, 0.0, 0.0)
        with self.assertRaises(MediaError):  # then the router sends it to fal
            v2.run("vertex:gemini-nano-banana-2.1", {"prompt": "x"})

    def test_a_regional_location_uses_its_own_host(self):
        v = Vertex(credentials={}, project="p", location="us-central1", token="t", http=lambda *a, **k: {})
        self.assertTrue(v.url("m").startswith("https://us-central1-aiplatform.googleapis.com/v1/projects/p/"
                                              "locations/us-central1/"))


class VertexKeyTests(unittest.TestCase):
    def test_a_google_cloud_api_key_signs_in_by_header_and_finds_its_project(self):
        from production.vertex import VertexKey

        calls = []

        def http(url, method="GET", headers=None, body=None, timeout=0, raw=False):
            calls.append((url, headers))
            if "no-such-model" in url:  # the free probe: Google names the key's project in its error
                raise MediaError("POST x: HTTP 404: Publisher model `projects/525/locations/us-central1/"
                                 "publishers/google/models/no-such-model` was not found")
            return answer_picture()

        v = VertexKey(key="k123", project="", http=http)
        v.run("vertex:gemini-nano-banana-2.1", {"prompt": "x"})
        v.run("vertex:gemini-nano-banana-2.1", {"prompt": "y"})
        self.assertEqual(len([u for u, _ in calls if "no-such-model" in u]), 1)  # asked once
        url, headers = calls[1]
        self.assertEqual(url, "https://aiplatform.googleapis.com/v1/projects/525/locations/global/publishers/"
                              "google/models/gemini-nano-banana-2.1:generateContent")
        self.assertEqual(headers["x-goog-api-key"], "k123")
        self.assertNotIn("k123", url)  # never in the address (it would show in logs)
        self.assertNotIn("Authorization", headers)

        def silent(url, method="GET", headers=None, body=None, timeout=0, raw=False):
            if "no-such-model" in url:
                raise MediaError("HTTP 404: not found")
            return answer_picture()
        nameless = VertexKey(key="k", project="", http=silent)  # no project learnt: the plain address
        self.assertEqual(nameless.url("m"), "https://aiplatform.googleapis.com/v1/publishers/google/models/m:"
                                            "generateContent")

    def test_a_retired_checker_model_gives_way_to_a_newer_one(self):
        from production.vertex import VertexKey

        asked = []

        def http(url, method="GET", headers=None, body=None, timeout=0, raw=False):
            model = url.rsplit("/", 1)[1].split(":")[0]
            asked.append(model)
            if model != "gemini-3.5-flash":
                raise MediaError(f"HTTP 404: Publisher model {model} was not found")
            return {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}

        v = VertexKey(key="k", project="p", http=http)
        self.assertEqual(v.run("vertex:gemini-2.5-flash", {"prompt": "x"}), {"output": "OK"})
        v.run("vertex:gemini-2.5-flash", {"prompt": "again"})
        self.assertEqual(asked[-1], "gemini-3.5-flash")  # remembered: straight to the one that answers
        self.assertEqual(asked.count("gemini-2.5-flash"), 1)

    def test_the_api_key_is_preferred_to_a_service_account(self):
        import os
        from unittest import mock

        from production import vertex

        with mock.patch.dict(os.environ, {"GEMINI_PLATFORM_API_KEY": "k", "GCP_PROJECT": ""}, clear=False):
            self.assertTrue(vertex.available())
            self.assertIsInstance(vertex.client(), vertex.VertexKey)


class ProvidersTests(unittest.TestCase):
    class Fal:
        def __init__(self):
            self.asked = []

        def run(self, endpoint, args):
            self.asked.append(endpoint)
            return {"images": [{"url": "https://fal/x.png"}]}

    class Broken:
        def run(self, endpoint, args):
            raise MediaError("HTTP 429 quota")

    def test_vertex_models_go_to_google_and_a_failure_back_to_the_fal_model_it_replaced(self):
        fal = self.Fal()
        fallback = {"vertex:nb": ["google/nano-banana-2.1/edit", "google/nano-banana-2.1"]}
        p = Providers(fal, self.Broken(), fallback)
        p.run("vertex:nb", {"prompt": "x", "image_urls": ["data:image/png;base64,AA"]})
        p.run("vertex:nb", {"prompt": "x"})
        p.run("fal-ai/other", {})
        self.assertEqual(fal.asked, ["google/nano-banana-2.1/edit", "google/nano-banana-2.1", "fal-ai/other"])
        self.assertEqual(p.fell_back, 2)

    def test_the_config_moves_nano_banana_2_1_and_the_checker_to_vertex(self):
        from story_factory.config import DEFAULTS

        cfg, fallback = use_vertex(DEFAULTS["stills"])
        self.assertEqual(cfg["models"]["shot"], "vertex:gemini-nano-banana-2.1")
        self.assertEqual(cfg["models"]["vision"], "vertex:gemini-2.5-flash")
        self.assertEqual(sorted(fallback["vertex:gemini-nano-banana-2.1"]),
                         ["google/nano-banana-2.1", "google/nano-banana-2.1/edit"])
        self.assertEqual(cfg["prices"]["vertex:gemini-nano-banana-2.1"], 0.055)
        self.assertEqual(DEFAULTS["stills"]["models"]["shot"], "google/nano-banana-2.1/edit")  # not changed in place
        off, none = use_vertex(dict(DEFAULTS["stills"], vertex={"enabled": False}))
        self.assertEqual((off["models"]["shot"], none), ("google/nano-banana-2.1/edit", {}))


if __name__ == "__main__":
    unittest.main()
