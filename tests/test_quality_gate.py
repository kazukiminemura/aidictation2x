from src.quality_gate import QualityGate


def test_change_ratio_zero_for_same_text() -> None:
    ratio = QualityGate.change_ratio("同じ文", "同じ文")
    assert ratio == 0.0


def test_quality_gate_accepts_small_change() -> None:
    gate = QualityGate([])
    result = gate.validate("今日は晴れ", "今日は晴れです", 0.6)
    assert result.accepted is True


def test_quality_gate_rejects_empty_output() -> None:
    gate = QualityGate([])
    result = gate.validate("入力", "", 0.5)
    assert result.accepted is False
    assert result.reason == "empty_output"
