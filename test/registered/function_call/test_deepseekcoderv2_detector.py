import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekcoderv2_detector import (
    DeepSeekCoderV2Detector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "stage-a-test-cpu")


def _make_tool(name, properties=None, required=None):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": properties
                or {"city": {"type": "string", "description": "City name"}},
                "required": required or ["city"],
            },
        ),
    )


def _wrap(func_name, python_call):
    return (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        f"{func_name}\n```python\n{python_call}\n```<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )


class TestDeepSeekCoderV2DetectorNonStreaming(unittest.TestCase):
    def setUp(self):
        self.tools = [
            _make_tool("get_weather"),
            _make_tool(
                "search",
                properties={
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["query"],
            ),
        ]
        self.detector = DeepSeekCoderV2Detector()

    def test_single_positional_call(self):
        text = _wrap("get_weather", 'get_weather("Hanoi")')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Hanoi"})

    def test_multiple_positional_args(self):
        text = _wrap("search", 'search("weather report", 5)')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"query": "weather report", "limit": 5},
        )

    def test_keyword_args_supported_too(self):
        text = _wrap("get_weather", 'get_weather(city="Paris")')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})

    def test_normal_text_before_tool_call(self):
        text = "Let me check the weather." + _wrap(
            "get_weather", 'get_weather("Tokyo")'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Let me check the weather.")
        self.assertEqual(len(result.calls), 1)

    def test_no_tool_call(self):
        text = "Just a normal response."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_has_tool_call(self):
        self.assertTrue(self.detector.has_tool_call("<｜tool▁calls▁begin｜>stuff"))
        self.assertFalse(self.detector.has_tool_call("no markers here"))

    def test_undefined_function_skipped(self):
        text = _wrap("unknown_fn", 'unknown_fn("x")')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)


class TestDeepSeekCoderV2DetectorStreaming(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.detector = DeepSeekCoderV2Detector()

    def test_streaming_full_block_in_chunks(self):
        text = _wrap("get_weather", 'get_weather("Hanoi")')
        # split into a few arbitrary chunks
        mid = len(text) // 2
        chunks = [text[:mid], text[mid:]]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tc in result.calls:
                while len(tool_calls) <= tc.tool_index:
                    tool_calls.append({"name": "", "parameters": ""})
                if tc.name:
                    tool_calls[tc.tool_index]["name"] = tc.name
                if tc.parameters:
                    tool_calls[tc.tool_index]["parameters"] += tc.parameters

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(json.loads(tool_calls[0]["parameters"]), {"city": "Hanoi"})


if __name__ == "__main__":
    unittest.main()
