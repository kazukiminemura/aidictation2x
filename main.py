import logging
import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from src.audio_capture import AudioConfig
from src.config_loader import load_json
from src.personal_dictionary import PersonalDictionary
from src.storage import Storage
from src.ui_app import build_app


APP_NAME = "AIDictation2x"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _bundle_root_dir() -> Path:
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _runtime_root_dir(bundle_root: Path) -> Path:
    if not _is_frozen():
        return bundle_root

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        runtime_root = Path(local_app_data) / APP_NAME
    else:
        runtime_root = Path.home() / "AppData" / "Local" / APP_NAME
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root


def _copy_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _prepare_runtime_files(bundle_root: Path, runtime_root: Path) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "config").mkdir(parents=True, exist_ok=True)
    (runtime_root / "data").mkdir(parents=True, exist_ok=True)
    (runtime_root / "models").mkdir(parents=True, exist_ok=True)

    if not _is_frozen():
        return

    _copy_if_missing(bundle_root / "config" / "app_settings.json", runtime_root / "config" / "app_settings.json")
    _copy_if_missing(bundle_root / "config" / "text_rules.json", runtime_root / "config" / "text_rules.json")
    _copy_if_missing(
        bundle_root / "config" / "personal_dictionary.json",
        runtime_root / "config" / "personal_dictionary.json",
    )
    _copy_if_missing(
        bundle_root / "config" / "llm_postedit_rules.json",
        runtime_root / "config" / "llm_postedit_rules.json",
    )


def _resolve_runtime_path(runtime_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return runtime_root / path


def _resolve_model_path(runtime_root: Path, raw_value: str) -> str:
    path = Path(raw_value)
    if path.is_absolute():
        return str(path)

    candidate = runtime_root / path
    if candidate.exists():
        return str(candidate)

    normalized = raw_value.replace("\\", "/")
    if normalized.startswith(("models/", "data/", "config/", "./", "../")):
        return str(candidate)
    if "\\" in raw_value:
        return str(candidate)
    return raw_value


def main() -> None:
    bundle_root = _bundle_root_dir()
    runtime_root = _runtime_root_dir(bundle_root)
    _prepare_runtime_files(bundle_root=bundle_root, runtime_root=runtime_root)

    settings = load_json(runtime_root / "config" / "app_settings.json")
    setup_logging(settings.get("log_level", "INFO"))

    rules_path = _resolve_runtime_path(runtime_root, settings.get("text_rules_file", "config/text_rules.json"))
    if not rules_path.exists():
        rules_path = bundle_root / settings.get("text_rules_file", "config/text_rules.json")
    rules = load_json(rules_path)

    audio_config = AudioConfig(
        sample_rate_hz=int(settings.get("sample_rate_hz", 16000)),
        channels=int(settings.get("channels", 1)),
    )

    storage = Storage(
        history_file=_resolve_runtime_path(runtime_root, settings.get("history_file", "data/history.json")),
        autosave_file=_resolve_runtime_path(runtime_root, settings.get("autosave_file", "data/last_session.json")),
        max_items=int(settings.get("max_history_items", 10)),
    )

    asr_defaults = {
        "backend": str(settings.get("asr_backend", "vosk")),
        "vosk_model_dir": str(
            _resolve_runtime_path(runtime_root, str(settings.get("vosk_model_dir", "models/vosk-model-ja")))
        ),
        "whisper_model_name": str(settings.get("whisper_model_name", "OpenVINO/whisper-large-v3-int8-ov")),
        "whisper_device": str(settings.get("whisper_device", "auto")),
        "whisper_compute_type": str(settings.get("whisper_compute_type", "int8")),
        "whisper_download_dir": str(
            _resolve_runtime_path(runtime_root, str(settings.get("whisper_download_dir", "models/whisper")))
        ),
    }
    personal_dictionary = PersonalDictionary(
        _resolve_runtime_path(runtime_root, settings.get("personal_dictionary_file", "config/personal_dictionary.json"))
    )

    llm_defaults = {
        "enabled": bool(settings.get("llm_enabled", True)),
        "model_path": _resolve_model_path(
            runtime_root,
            str(settings.get("llm_model_path", "OpenVINO/Qwen3-8B-int4-cw-ov")),
        ),
        "strength": str(settings.get("llm_strength", "medium")),
        "max_input_chars": int(settings.get("llm_max_input_chars", 1200)),
        "max_change_ratio": float(settings.get("llm_max_change_ratio", 0.35)),
        "domain_hint": str(settings.get("llm_domain_hint", "")),
        "timeout_ms": int(settings.get("llm_timeout_ms", 8000)),
        "blocked_patterns": list(settings.get("llm_blocked_patterns", [])),
        "device": str(settings.get("llm_device", "GPU")),
        "auto_download": bool(settings.get("llm_auto_download", False)),
        "download_dir": str(
            _resolve_runtime_path(runtime_root, str(settings.get("llm_download_dir", "models/openvino")))
        ),
    }

    root = tk.Tk()
    try:
        build_app(
            root=root,
            audio_config=audio_config,
            storage=storage,
            rules=rules,
            personal_dictionary=personal_dictionary,
            enable_system_wide_input_default=bool(settings.get("enable_system_wide_input", True)),
            llm_defaults=llm_defaults,
            asr_defaults=asr_defaults,
            root_dir=runtime_root,
        )
    except FileNotFoundError as exc:
        root.withdraw()
        messagebox.showerror("Model not found", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
