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
