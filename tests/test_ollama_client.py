from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.llm.ollama_client import OllamaCloudClient, OllamaCloudError, _safe_json_parse
from src.utils.cost_tracker import CostTracker


def _fake_response(text="مرحبا", prompt_tokens=10, completion_tokens=5, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "error body"
    resp.json.return_value = {
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }
    return resp


@patch("src.llm.ollama_client.requests.post")
def test_chat_returns_text_and_tracks_cost(mock_post):
    mock_post.return_value = _fake_response("أهلاً بك")
    tracker = CostTracker()
    client = OllamaCloudClient(cost_tracker=tracker)

    response = client.chat([{"role": "user", "content": "مرحبا"}], purpose="test")

    assert response.text == "أهلاً بك"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert tracker.total_prompt_tokens == 10
    assert len(tracker.calls) == 1


@patch("src.llm.ollama_client.requests.post")
def test_chat_retries_then_raises_on_persistent_failure(mock_post):
    mock_post.side_effect = ConnectionError("boom")
    client = OllamaCloudClient()
    client.max_retries = 2

    try:
        client.chat([{"role": "user", "content": "x"}])
        assert False, "expected OllamaCloudError"
    except OllamaCloudError:
        pass
    assert mock_post.call_count == 2


@patch("src.llm.ollama_client.requests.post")
def test_chat_raises_on_http_error_status(mock_post):
    mock_post.return_value = _fake_response(status_code=500)
    client = OllamaCloudClient()
    client.max_retries = 1

    try:
        client.chat([{"role": "user", "content": "x"}])
        assert False, "expected OllamaCloudError"
    except OllamaCloudError:
        pass


def test_safe_json_parse_handles_code_fences():
    text = '```json\n{"a": 1, "b": "نص"}\n```'
    parsed = _safe_json_parse(text)
    assert parsed == {"a": 1, "b": "نص"}


def test_safe_json_parse_handles_leading_trailing_text():
    text = 'Sure, here is the JSON: {"score": 0.8} Hope that helps!'
    parsed = _safe_json_parse(text)
    assert parsed["score"] == 0.8


def test_safe_json_parse_returns_empty_dict_on_garbage():
    assert _safe_json_parse("not json at all") == {}


@patch("src.llm.ollama_client.requests.post")
def test_complete_json_uses_fast_model_by_default(mock_post):
    mock_post.return_value = _fake_response('{"route": "basic"}')
    client = OllamaCloudClient()
    result = client.complete_json("السؤال", purpose="routing")
    assert result == {"route": "basic"}
