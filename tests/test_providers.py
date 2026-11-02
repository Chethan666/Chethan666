"""Provider clients exercised through the real SDKs with a mocked network.

Skipped when the `anthropic` / `openai` packages aren't installed.
"""

import importlib.util
import json
import os
import unittest

from story_factory import db, providers
from story_factory.llm import LLMError, ModelSpec, UsageRecorder
from tests.fakes import test_config

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
HAS_OPENAI = importlib.util.find_spec("openai") is not None
# Both SDKs send traffic through httpx2, the maintained fork of httpx.
HAS_ANTHROPIC = HAS_ANTHROPIC and importlib.util.find_spec("httpx2") is not None
HAS_OPENAI = HAS_OPENAI and importlib.util.find_spec("httpx2") is not None


def sse(events: list[tuple[str, dict]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def claude_stream(text: str, stop_reason: str = "end_turn") -> bytes:
    return sse([
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-fable-5-1",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 7}}),
        ("message_stop", {"type": "message_stop"}),
    ])


@unittest.skipUnless(HAS_ANTHROPIC, "anthropic SDK not installed")
class AnthropicClientTests(unittest.TestCase):
    def make(self, responses: list[bytes], model: str = "claude-fable-5-1@high"):
        import anthropic
        import httpx2 as httpx

        self.requests: list[dict] = []
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append({"url": str(request.url), "headers": dict(request.headers),
                                  "body": json.loads(request.content)})
            return httpx.Response(200, content=queue.pop(0), headers={"content-type": "text/event-stream"})

        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
        self.conn = db.connect(":memory:")
        client = providers.AnthropicClient(ModelSpec.parse(f"anthropic:{model}"), test_config(),
                                           UsageRecorder(self.conn, test_config()))
        client.client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(
            transport=httpx.MockTransport(handler)))
        return client

    def test_structured_output_request_and_parse(self):
        client = self.make([claude_stream('{"winner": "A", "reason": "tighter"}')])
        result = client.generate_json("compare", "system text", "prompt text",
                                      {"type": "object", "properties": {}, "additionalProperties": False})
        self.assertEqual(result["winner"], "A")
        body = self.requests[0]["body"]
        self.assertEqual(body["model"], "claude-fable-5-1")
        self.assertEqual(body["output_config"]["effort"], "high")
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(body["fallbacks"], "default")
        self.assertNotIn("temperature", body)
        self.assertIn("server-side-fallback-2026-07-01", self.requests[0]["headers"]["anthropic-beta"])
        cost = self.conn.execute("SELECT task, input_units, output_units FROM costs").fetchone()
        self.assertEqual(tuple(cost), ("compare", 11, 7))

    def test_refusal_and_truncation_raise(self):
        client = self.make([claude_stream("", "refusal"), claude_stream('{"a"', "max_tokens")])
        with self.assertRaises(LLMError):
            client.generate_json("t", "s", "p", {"type": "object"})
        with self.assertRaises(LLMError):
            client.generate_json("t", "s", "p", {"type": "object"})

    def test_research_uses_web_search_and_resumes_paused_turns(self):
        client = self.make([claude_stream("partial", "pause_turn"), claude_stream("Final notes.")],
                           model="claude-opus-5-5@high")
        self.assertEqual(client.research("dossier_research", "s", "p"), "Final notes.")
        first, second = (r["body"] for r in self.requests)
        self.assertEqual(first["tools"][0]["type"], "web_search_20260209")
        self.assertEqual(second["messages"][-1]["role"], "assistant")


@unittest.skipUnless(HAS_OPENAI, "openai SDK not installed")
class OpenAIClientTests(unittest.TestCase):
    def test_strict_schema_request_and_parse(self):
        import httpx2 as httpx
        import openai

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-5.5",
                "status": "completed", "output": [{
                    "type": "message", "id": "m1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": '{"winner": "B", "reason": "x"}',
                                 "annotations": []}]}],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}},
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": []})

        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        conn = db.connect(":memory:")
        client = providers.OpenAIClient(ModelSpec.parse("openai:gpt-5.5@high"), test_config(),
                                        UsageRecorder(conn, test_config()))
        client.client = openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        schema = {"type": "object", "properties": {"winner": {"type": "string"}}, "required": ["winner"],
                  "additionalProperties": False}
        self.assertEqual(client.generate_json("compare", "sys", "prompt", schema)["winner"], "B")
        body = captured["body"]
        self.assertEqual(body["text"]["format"], {"type": "json_schema", "name": "compare", "schema": schema,
                                                  "strict": True})
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(body["instructions"], "sys")


if __name__ == "__main__":
    unittest.main()
