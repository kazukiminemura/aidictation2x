import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(default, fp, ensure_ascii=False, indent=2)
        return dict(default)
    with path.open("r", encoding="utf-8-sig") as fp:
        return json.load(fp)
