import json
import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from src.audio_capture import AudioConfig
from src.config_loader import load_json
from src.personal_dictionary import PersonalDictionary
from src.storage import Storage
from src.ui_app import build_app


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    settings = load_json(root_dir / "config" / "app_settings.json")
    setup_logging(settings.get("log_level", "INFO"))

    rules_path = root_dir / settings.get("text_rules_file", "config/text_rules.json")
    rules = load_json(rules_path)

    audio_config = AudioConfig(
        sample_rate_hz=int(settings.get("sample_rate_hz", 16000)),
        channels=int(settings.get("channels", 1)),
    )

    storage = Storage(
        history_file=root_dir / settings.get("history_file", "data/history.json"),
        autosave_file=root_dir / settings.get("autosave_file", "data/last_session.json"),
        max_items=int(settings.get("max_history_items", 10)),
    )

    model_dir = root_dir / settings.get("vosk_model_dir", "models/vosk-model-ja")
    personal_dictionary = PersonalDictionary(
        root_dir / settings.get("personal_dictionary_file", "config/personal_dictionary.json")
    )

    root = tk.Tk()
    try:
        build_app(
            root=root,
            model_dir=model_dir,
            audio_config=audio_config,
            storage=storage,
            rules=rules,
            personal_dictionary=personal_dictionary,
            enable_system_wide_input_default=bool(settings.get("enable_system_wide_input", True)),
        )
    except FileNotFoundError as exc:
        root.withdraw()
        messagebox.showerror("モデル未配置", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
