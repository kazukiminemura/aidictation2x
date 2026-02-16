import json
from pathlib import Path

from src.storage import Storage


def test_storage_loads_legacy_history_without_llm_fields(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    autosave_path = tmp_path / "autosave.json"
    history_path.write_text(
        json.dumps([
            {
                "timestamp": "2026-02-16T10:00:00",
                "raw_text": "raw",
                "final_text": "final"
            }
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    storage = Storage(history_file=history_path, autosave_file=autosave_path, max_items=10)
    items = storage.load_history()

    assert len(items) == 1
    assert items[0].llm_applied is False
    assert items[0].llm_latency_ms == 0
    assert items[0].fallback_reason == ""
    assert items[0].processing_total_ms == 0
    assert items[0].processing_breakdown_ms == {}


def test_storage_saves_llm_metadata(tmp_path: Path) -> None:
    storage = Storage(
        history_file=tmp_path / "history.json",
        autosave_file=tmp_path / "autosave.json",
        max_items=10,
    )

    storage.append_history(
        raw_text="raw",
        final_text="final",
        llm_applied=True,
        llm_latency_ms=123,
        fallback_reason="",
    )

    items = storage.load_history()
    assert items[0].llm_applied is True
    assert items[0].llm_latency_ms == 123
    assert items[0].processing_total_ms == 0
    assert items[0].processing_breakdown_ms == {}


def test_storage_saves_processing_timing_metadata(tmp_path: Path) -> None:
    storage = Storage(
        history_file=tmp_path / "history.json",
        autosave_file=tmp_path / "autosave.json",
        max_items=10,
    )

    storage.append_history(
        raw_text="raw",
        final_text="final",
        llm_applied=True,
        llm_latency_ms=123,
        fallback_reason="",
        processing_total_ms=456,
        processing_breakdown_ms={"asr": 100, "rules": 120, "llm": 200, "storage": 36},
    )

    items = storage.load_history()
    assert items[0].processing_total_ms == 456
    assert items[0].processing_breakdown_ms["asr"] == 100
    assert items[0].processing_breakdown_ms["llm"] == 200
