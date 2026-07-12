import ast
import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class DeepSeekCoderV2Detector(BaseFormatDetector):
    """
    Detector for DeepSeek-Coder-V2 (and DeepSeek-V2-Lite) tool call format.

    These checkpoints reuse the DeepSeek-V3 wrapper tokens (pretraining
    exposure), but emit the call arguments as a Python function call instead
    of a JSON object, and use positional args instead of keyword args:

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{function_name}
    ```python
    {function_name}({positional_args})
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    ```
    Example:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```python
    get_weather("Hanoi")
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    ```

    Positional arguments are mapped back onto the tool's declared parameter
    names (from the request's JSON schema `properties`, in declaration order)
    since the OpenAI `tool_calls[].function.arguments` format expects a JSON
    object. Keyword arguments (`city="Hanoi"`), if the model emits them, are
    used as-is.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```python\n(.*)\n```<｜tool▁call▁end｜>"
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _positional_args_to_kwargs(self, call: ast.Call, tool_index_map, func_name):
        """Map a parsed ast.Call's positional + keyword args to a kwargs dict."""
        arguments = {}

        param_names: List[str] = []
        tool = tool_index_map.get(func_name)
        if tool is not None:
            parameters = getattr(tool.function, "parameters", None) or {}
            properties = (
                parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            )
            param_names = list(properties.keys())

        for i, arg in enumerate(call.args):
            value = self._get_parameter_value(arg)
            if i < len(param_names):
                arguments[param_names[i]] = value
            else:
                # More positional args than declared parameters: best effort.
                arguments[f"arg{i}"] = value

        for keyword in call.keywords:
            if keyword.arg is not None:
                arguments[keyword.arg] = self._get_parameter_value(keyword.value)

        return arguments

    def _get_parameter_value(self, val):
        if isinstance(val, ast.Constant):
            return val.value
        elif isinstance(val, ast.Dict):
            return {
                k.value: self._get_parameter_value(v)
                for k, v in zip(val.keys, val.values)
            }
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        elif isinstance(val, ast.Tuple):
            return [self._get_parameter_value(v) for v in val.elts]
        else:
            raise ValueError("Tool call arguments must be literals")

    def _parse_python_call(self, func_args_raw: str, func_name: str, tools: List[Tool]):
        tool_by_name = {t.function.name: t for t in tools if t.function.name}
        module = ast.parse(func_args_raw.strip())
        call = module.body[0].value
        if not isinstance(call, ast.Call):
            raise ValueError("Expected a single function call expression")
        return self._positional_args_to_kwargs(call, tool_by_name, func_name)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        tool_indices = self._get_tool_indices(tools)
        try:
            for call_index, match_result in enumerate(match_result_list):
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                if func_detail is None:
                    continue
                func_name = func_detail.group(2).strip()
                func_args_raw = func_detail.group(3)

                if func_name not in tool_indices:
                    logger.warning(
                        f"Model attempted to call undefined function: {func_name}"
                    )
                    continue

                arguments = self._parse_python_call(func_args_raw, func_name, tools)
                calls.append(
                    ToolCallItem(
                        tool_index=call_index,
                        name=func_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing. Since Python-literal call arguments
        can't be diffed the way partial JSON can, arguments are buffered and
        emitted in one shot per tool call, once its ```python ... ``` block
        closes (only the tool name is streamed early).
        """
        self._buffer += new_text
        current_text = self._buffer

        has_tool_call = (
            self.bot_token in current_text or "<｜tool▁call▁begin｜>" in current_text
        )

        if not has_tool_call:
            self._buffer = ""
            for e_token in [self.eot_token, "```", "<｜tool▁call▁end｜>"]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: List[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```python\n(.*)\n```.*",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(2).strip()

                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    # Full ```python block for this call is present: complete it in one shot.
                    call_end_pattern = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
                    match = re.search(call_end_pattern, current_text, re.DOTALL)
                    if match:
                        func_detail = re.search(
                            self.func_detail_regex, match.group(0), re.DOTALL
                        )
                        if func_detail is not None:
                            func_args_raw = func_detail.group(3)
                            try:
                                arguments = self._parse_python_call(
                                    func_args_raw, func_name, tools
                                )
                            except Exception:
                                arguments = {}
                            args_json = json.dumps(arguments, ensure_ascii=False)

                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=args_json,
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                args_json
                            )
                            self.prev_tool_call_arr[self.current_tool_id] = {
                                "name": func_name,
                                "arguments": arguments,
                            }

                        self._buffer = current_text[match.end() :]
                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
            + name
            + "\n```python\n",
            end="\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            trigger="<｜tool▁calls▁begin｜>",
        )
