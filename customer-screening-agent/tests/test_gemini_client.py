"""Gemini REST 客户端契约测试。

不访问真实网络，而是模拟 Gemini 的 HTTP 响应，验证最容易出错的
请求协议字段和结构化 JSON 解析。真实模型质量仍需配置 API Key 单独验收。
"""

import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.llm_client import GeminiLLMClient
from backend.models import Intent, LLMJudgement


def _response_payload():
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"intent":"need_more_info",'
                                '"emotion_negative":false,"confidence":0.9,'
                                '"draft_reply":"我来为您介绍。",'
                                '"reasoning":"客户在询问产品信息"}'
                            )
                        }
                    ]
                }
            }
        ]
    }


def test_gemini_classify_uses_rest_camel_case_and_parses_response():
    client = GeminiLLMClient("test-key", "test-model")
    response = Mock()
    response.json.return_value = _response_payload()

    with patch("backend.llm_client.httpx.post", return_value=response) as post:
        result = client.classify([], "这个产品怎么收费？", temperature=0.5, perspective="检查语气")

    response.raise_for_status.assert_called_once_with()
    assert result.intent == Intent.NEED_MORE_INFO
    assert result.emotion_negative is False

    payload = post.call_args.kwargs["json"]
    assert "systemInstruction" in payload
    assert "system_instruction" not in payload
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseSchema" in payload["generationConfig"]
    assert "response_mime_type" not in payload["generationConfig"]
    assert "response_schema" not in payload["generationConfig"]
    assert payload["contents"][0]["role"] == "user"
    assert "这个产品怎么收费？" in payload["contents"][0]["parts"][0]["text"]


def test_gemini_review_uses_same_structured_output_contract():
    client = GeminiLLMClient("test-key", "test-model")
    response = Mock()
    response.json.return_value = _response_payload()

    with patch("backend.llm_client.httpx.post", return_value=response) as post:
        result = client.review([], "请介绍产品", [
            LLMJudgement(
                intent=Intent.NEED_MORE_INFO,
                emotion_negative=False,
                confidence=0.8,
                draft_reply="",
            )
        ])

    assert result.intent == Intent.NEED_MORE_INFO
    payload = post.call_args.kwargs["json"]
    assert "systemInstruction" in payload
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseSchema" in payload["generationConfig"]
