import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DictionaryEntry:
    reading: str
    surface: str
    count: int = 1


class PersonalDictionary:
    def __init__(self, dictionary_file: Path):
        self.dictionary_file = dictionary_file
        self.dictionary_file.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, DictionaryEntry] = {}
        self.load()

    def load(self) -> None:
        if not self.dictionary_file.exists():
            self.save()
            return
        with self.dictionary_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        entries = payload.get("entries", [])
        self._entries = {}
        for item in entries:
            entry = DictionaryEntry(**item)
            if entry.reading:
                self._entries[entry.reading] = entry

    def save(self) -> None:
        serializable = {
            "entries": [asdict(item) for item in self.list_entries()],
        }
        with self.dictionary_file.open("w", encoding="utf-8") as fp:
            json.dump(serializable, fp, ensure_ascii=False, indent=2)

    def list_entries(self) -> List[DictionaryEntry]:
        return sorted(self._entries.values(), key=lambda x: (-len(x.reading), x.reading))

    def add_or_update(self, reading: str, surface: str) -> None:
        reading = reading.strip()
        surface = surface.strip()
        if not reading or not surface:
            raise ValueError("読みと表記の両方を入力してください。")
        if reading in self._entries:
            current = self._entries[reading]
            current.surface = surface
            current.count += 1
        else:
            self._entries[reading] = DictionaryEntry(reading=reading, surface=surface, count=1)
        self.save()

    def remove(self, reading: str) -> None:
        if reading in self._entries:
            del self._entries[reading]
            self.save()

    def apply(self, text: str) -> str:
        output = text
        for entry in self.list_entries():
            output = output.replace(entry.reading, entry.surface)
        return output
