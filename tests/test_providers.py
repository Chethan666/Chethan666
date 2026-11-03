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


@unittest.skipUnless(HAS_OPENAI, "openai SDK not installed")
class XAIClientTests(unittest.TestCase):
    def make(self, responses):
        import httpx2 as httpx
        import openai

        self.bodies: list[dict] = []
        queue = list(responses)

        def handler(request):
            self.bodies.append(json.loads(request.content))
            self.url = str(request.url)
            status, payload = queue.pop(0)
            return httpx.Response(status, json=payload)

        os.environ.setdefault("XAI_API_KEY", "test-key")
        client = providers.XAIClient(ModelSpec.parse("xai:grok-4.7"), test_config(),
                                     UsageRecorder(db.connect(":memory:"), test_config()))
        client.client = openai.OpenAI(api_key="k", base_url=providers.XAIClient.BASE_URL,
                                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        return client

    @staticmethod
    def completion(text: str) -> dict:
        return {"id": "c1", "object": "chat.completion", "created": 0, "model": "grok-4.7",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}

    def test_strict_schema_then_fallback_to_json_mode(self):
        client = self.make([(200, self.completion('{"winner": "A"}'))])
        self.assertEqual(client.generate_json("t", "sys", "p", {"type": "object"})["winner"], "A")
        self.assertTrue(self.url.startswith("https://api.x.ai/v1/chat/completions"))
        self.assertEqual(self.bodies[0]["response_format"]["type"], "json_schema")
        self.assertEqual(self.bodies[0]["messages"][0], {"role": "system", "content": "sys"})

        client = self.make([(400, {"error": {"message": "schema not supported"}}),
                            (200, self.completion('{"winner": "B"}'))])
        self.assertEqual(client.generate_json("t", "sys", "p", {"type": "object"})["winner"], "B")
        self.assertEqual(self.bodies[1]["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", self.bodies[1]["messages"][1]["content"])


@unittest.skipUnless(HAS_ANTHROPIC and HAS_OPENAI, "anthropic and openai SDKs not installed")
class AWSClientTests(unittest.TestCase):
    def setUp(self):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "bedrock-test-key"
        os.environ.pop("AWS_REGION", None)
        os.environ.pop("AWS_DEFAULT_REGION", None)
        self.cfg = test_config()
        self.recorder = UsageRecorder(db.connect(":memory:"), self.cfg)
        self.requests: list = []

    def transport(self, responses):
        import httpx2 as httpx

        queue = list(responses)

        def handler(request):
            self.requests.append(request)
            status, content, kind = queue.pop(0)
            return httpx.Response(status, content=content, headers={"content-type": kind})

        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_build_picks_the_right_client(self):
        spec = lambda text: ModelSpec.parse(text)
        self.assertIsInstance(providers.build(spec("aws:anthropic.claude-opus-5-5"), self.cfg, self.recorder),
                              providers.AWSClaudeClient)
        self.assertIsInstance(providers.build(spec("aws:openai.gpt-5.5"), self.cfg, self.recorder),
                              providers.AWSResponsesClient)
        self.assertIsInstance(providers.build(spec("aws:xai.grok-4.6"), self.cfg, self.recorder),
                              providers.AWSChatClient)
        self.assertEqual(spec("aws:xai.grok-4.6").family, "xai")

    def test_claude_on_bedrock(self):
        import anthropic

        client = providers.build(ModelSpec.parse("aws:anthropic.claude-opus-5-5@high"), self.cfg, self.recorder)
        self.assertFalse(client.supports_web_search)
        client.client = anthropic.AnthropicBedrockMantle(
            api_key="bedrock-test-key", aws_region="us-east-1",
            http_client=self.transport([(200, claude_stream('{"ok": true}'), "text/event-stream")]))
        self.assertEqual(client.generate_json("t", "s", "p", {"type": "object"}), {"ok": True})
        request = self.requests[0]
        body = json.loads(request.content)
        self.assertEqual(request.url.host, "bedrock-mantle.us-east-1.api.aws")
        self.assertTrue(request.url.path.endswith("/v1/messages"))
        self.assertEqual(body["model"], "anthropic.claude-opus-5-5")
        self.assertEqual(body["output_config"]["effort"], "high")
        self.assertNotIn("fallbacks", body)  # not offered on Bedrock
        with self.assertRaises(LLMError):
            client.research("r", "s", "p")

    def test_gpt_and_grok_on_bedrock(self):
        import openai

        gpt = providers.build(ModelSpec.parse("aws:openai.gpt-5.5@low"), self.cfg, self.recorder)
        self.assertEqual(str(gpt.client.base_url), "https://bedrock-mantle.us-east-1.api.aws/openai/v1/")
        grok = providers.build(ModelSpec.parse("aws:xai.grok-4.6"), self.cfg, self.recorder)
        grok.client = openai.OpenAI(api_key="k", base_url=str(grok.client.base_url), http_client=self.transport([
            (200, json.dumps(XAIClientTests.completion('{"winner": "A"}')).encode(), "application/json")]))
        self.assertEqual(grok.generate_json("t", "s", "p", {"type": "object"}), {"winner": "A"})
        self.assertEqual(self.requests[0].url.path, "/openai/v1/chat/completions")
        self.assertEqual(json.loads(self.requests[0].content)["model"], "xai.grok-4.6")

    def test_404_switches_to_the_other_bedrock_address(self):
        import openai

        grok = providers.build(ModelSpec.parse("aws:xai.grok-4.6"), self.cfg, self.recorder)
        ok = json.dumps(XAIClientTests.completion('{"winner": "B"}')).encode()
        grok.client = openai.OpenAI(api_key="k", base_url=str(grok.client.base_url), http_client=self.transport([
            (404, b'{"error": {"message": "not found"}}', "application/json"),
            (200, ok, "application/json"), (200, ok, "application/json")]))
        self.assertEqual(grok.generate_json("t", "s", "p", {"type": "object"}), {"winner": "B"})
        self.assertEqual([r.url.path for r in self.requests], ["/openai/v1/chat/completions", "/v1/chat/completions"])
        grok.generate_json("t", "s", "p", {"type": "object"})  # stays on the address that worked
        self.assertEqual(self.requests[-1].url.path, "/v1/chat/completions")

    def test_missing_bedrock_key_is_reported(self):
        del os.environ["AWS_BEARER_TOKEN_BEDROCK"]
        with self.assertRaises(LLMError):
            providers.build(ModelSpec.parse("aws:xai.grok-4.6"), self.cfg, self.recorder)


if __name__ == "__main__":
    unittest.main()
