import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class HistoryItem:
    timestamp: str
    raw_text: str
    final_text: str
    llm_applied: bool = False
    llm_latency_ms: int = 0
    fallback_reason: str = ""


class Storage:
    def __init__(self, history_file: Path, autosave_file: Path, max_items: int):
        self.history_file = history_file
        self.autosave_file = autosave_file
        self.max_items = max_items
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.autosave_file.parent.mkdir(parents=True, exist_ok=True)

    def save_autosave(
        self,
        raw_text: str,
        final_text: str,
        llm_applied: bool = False,
        llm_latency_ms: int = 0,
        fallback_reason: str = "",
    ) -> None:
        payload = asdict(
            HistoryItem(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                raw_text=raw_text,
                final_text=final_text,
                llm_applied=llm_applied,
                llm_latency_ms=llm_latency_ms,
                fallback_reason=fallback_reason,
            )
        )
        with self.autosave_file.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def load_autosave(self) -> HistoryItem | None:
        if not self.autosave_file.exists():
            return None
        with self.autosave_file.open("r", encoding="utf-8-sig") as fp:
            payload = json.load(fp)
        return self._to_history_item(payload)

    def append_history(
        self,
        raw_text: str,
        final_text: str,
        llm_applied: bool = False,
        llm_latency_ms: int = 0,
        fallback_reason: str = "",
    ) -> None:
        current = self.load_history()
        current.insert(
            0,
            HistoryItem(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                raw_text=raw_text,
                final_text=final_text,
                llm_applied=llm_applied,
                llm_latency_ms=llm_latency_ms,
                fallback_reason=fallback_reason,
            ),
        )
        current = current[: self.max_items]
        serializable = [asdict(item) for item in current]
        with self.history_file.open("w", encoding="utf-8") as fp:
            json.dump(serializable, fp, ensure_ascii=False, indent=2)

    def load_history(self) -> List[HistoryItem]:
        if not self.history_file.exists():
            return []
        with self.history_file.open("r", encoding="utf-8-sig") as fp:
            payload = json.load(fp)
        return [self._to_history_item(item) for item in payload]

    @staticmethod
    def _to_history_item(payload: dict) -> HistoryItem:
        return HistoryItem(
            timestamp=payload.get("timestamp", ""),
            raw_text=payload.get("raw_text", ""),
            final_text=payload.get("final_text", ""),
            llm_applied=bool(payload.get("llm_applied", False)),
            llm_latency_ms=int(payload.get("llm_latency_ms", 0)),
            fallback_reason=str(payload.get("fallback_reason", "")),
        )
