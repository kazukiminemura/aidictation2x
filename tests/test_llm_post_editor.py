from pathlib import Path

from src.llm_post_editor import LLMOptions, LLMPostEditor


class FakeBackend:
    def __init__(self, output: str):
        self.output = output

    def generate(self, text: str, options: LLMOptions, timeout_ms: int) -> str:  # noqa: ARG002
        return self.output


BASE_OPTIONS = LLMOptions(
    enabled=True,
    strength="medium",
    max_input_chars=1200,
    max_change_ratio=0.35,
    domain_hint="",
)


def test_refine_disabled_returns_input() -> None:
    editor = LLMPostEditor(model_path=Path("."), backend=FakeBackend("ignored"))
    result = editor.refine("raw", "前処理結果", LLMOptions(**{**BASE_OPTIONS.__dict__, "enabled": False}))

    assert result.final_text == "前処理結果"
    assert result.applied is False
    assert result.fallback_reason == "disabled"


def test_refine_change_ratio_exceeded_fallback() -> None:
    editor = LLMPostEditor(model_path=Path("."), backend=FakeBackend("x" * 100))
    result = editor.refine("raw", "短文", BASE_OPTIONS)

    assert result.applied is False
    assert result.fallback_reason == "change_ratio_exceeded"
    assert result.final_text == "短文"


def test_refine_blocked_pattern_fallback() -> None:
    editor = LLMPostEditor(
        model_path=Path("."),
        backend=FakeBackend("これはSECRETを含む文です"),
        blocked_patterns=[r"SECRET"],
    )
    result = editor.refine("raw", "これは文です", BASE_OPTIONS)

    assert result.applied is False
    assert result.fallback_reason == "blocked_pattern"


def test_refine_applies_with_edits() -> None:
    editor = LLMPostEditor(model_path=Path("."), backend=FakeBackend("正式な文章です。"))
    result = editor.refine("raw", "正式な分掌です。", BASE_OPTIONS)

    assert result.applied is True
    assert result.fallback_reason == ""
    assert result.final_text == "正式な文章です。"
    assert result.edits


def test_refine_splits_long_input() -> None:
    calls = []

    class TrackingBackend:
        def generate(self, text: str, options: LLMOptions, timeout_ms: int) -> str:  # noqa: ARG002
            calls.append(text)
            return text

    editor = LLMPostEditor(model_path=Path("."), backend=TrackingBackend())
    long_text = "。".join(["これはテスト文です" for _ in range(20)]) + "。"
    options = LLMOptions(**{**BASE_OPTIONS.__dict__, "max_input_chars": 40})
    result = editor.refine("raw", long_text, options)

    assert result.applied is True
    assert len(calls) > 1


def test_refine_uses_external_agent_with_raw_and_rest_response() -> None:
    calls: list[tuple[str, str, int]] = []

    def external_agent_caller(url: str, prompt: str, timeout_ms: int) -> tuple[str, str]:
        calls.append((url, prompt, timeout_ms))
        return "外部補正済みの文です。", '{"text":"外部補正済みの文です。"}'

    editor = LLMPostEditor(
        model_path=Path("."),
        backend=FakeBackend("内部補正結果"),
        external_agent_caller=external_agent_caller,
    )
    options = LLMOptions(
        **{
            **BASE_OPTIONS.__dict__,
            "external_agent_enabled": True,
            "external_agent_url": "http://127.0.0.1:8000/v1/agent/chat",
        }
    )
    result = editor.refine("音声認識の原文", "前処理済み文", options)

    assert result.applied is True
    assert result.final_text == "外部補正済みの文です。"
    assert result.external_agent_response == "外部補正済みの文です。"
    assert result.external_agent_raw_response == '{"text":"外部補正済みの文です。"}'
    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:8000/v1/agent/chat"
    assert calls[0][1] == "音声認識の原文"


def test_refine_external_agent_error_fallback() -> None:
    def external_agent_caller(url: str, prompt: str, timeout_ms: int) -> str:  # noqa: ARG001
        raise RuntimeError("external_agent_error")

    editor = LLMPostEditor(
        model_path=Path("."),
        backend=FakeBackend("内部補正結果"),
        external_agent_caller=external_agent_caller,
    )
    options = LLMOptions(
        **{
            **BASE_OPTIONS.__dict__,
            "external_agent_enabled": True,
            "external_agent_url": "http://127.0.0.1:8000/v1/agent/chat",
        }
    )
    result = editor.refine("音声認識の原文", "前処理済み文", options)

    assert result.applied is False
    assert result.fallback_reason == "external_agent_error"
    assert result.final_text == "前処理済み文"


def test_refine_external_agent_bypasses_change_ratio_gate() -> None:
    def external_agent_caller(url: str, prompt: str, timeout_ms: int) -> str:  # noqa: ARG001
        return "まったく別の長い応答テキストです。内容が大きく変わっていても外部連携時は採用される。"

    editor = LLMPostEditor(
        model_path=Path("."),
        backend=FakeBackend("内部補正結果"),
        external_agent_caller=external_agent_caller,
    )
    options = LLMOptions(
        **{
            **BASE_OPTIONS.__dict__,
            "external_agent_enabled": True,
            "external_agent_url": "http://127.0.0.1:8000/v1/agent/chat",
            "max_change_ratio": 0.01,
        }
    )
    result = editor.refine("原文", "短文", options)

    assert result.applied is True
    assert result.fallback_reason == ""
    assert "外部連携時は採用される" in result.final_text


def test_refine_external_agent_uses_minimum_timeout() -> None:
    captured: dict[str, int] = {"timeout_ms": 0}

    def external_agent_caller(url: str, prompt: str, timeout_ms: int) -> str:  # noqa: ARG001
        captured["timeout_ms"] = timeout_ms
        return "応答"

    editor = LLMPostEditor(
        model_path=Path("."),
        backend=FakeBackend("内部補正結果"),
        external_agent_caller=external_agent_caller,
        timeout_ms=8000,
    )
    options = LLMOptions(
        **{
            **BASE_OPTIONS.__dict__,
            "external_agent_enabled": True,
            "external_agent_url": "http://127.0.0.1:8000/v1/agent/chat",
        }
    )
    _ = editor.refine("原文", "前処理済み", options)

    assert captured["timeout_ms"] >= 300000
