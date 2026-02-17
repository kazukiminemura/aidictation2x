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


def setup_logging(level: str, runtime_root: Path | None = None) -> None:
    handlers: list[logging.Handler] = []
    if runtime_root is not None:
        log_dir = runtime_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "app.log", encoding="utf-8"))
    handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
        handlers=handlers,
    )


def _ensure_standard_streams() -> None:
    # Windowed executable can start without stdio; some libs (logging/tqdm) require writable streams.
    devnull_out = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    devnull_err = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = devnull_out
    if sys.stderr is None:
        sys.stderr = devnull_err
    if getattr(sys, "__stdout__", None) is None:
        sys.__stdout__ = sys.stdout
    if getattr(sys, "__stderr__", None) is None:
        sys.__stderr__ = sys.stderr


def _configure_hf_runtime_env() -> None:
    # Avoid progress / xet paths that can fail in windowed builds without a console stream.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _bundle_root_dir() -> Path:
    if _is_frozen():
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass).resolve()
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


def _config_source_dirs(bundle_root: Path) -> list[Path]:
    if not _is_frozen():
        return [bundle_root / "config"]

    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        bundle_root / "config",
        exe_dir / "config",
        exe_dir / "_internal" / "config",
    ]
    return [path for path in candidates if path.exists()]


def _copy_config_if_missing(config_name: str, runtime_root: Path, source_dirs: list[Path]) -> None:
    destination = runtime_root / "config" / config_name
    if destination.exists():
        return
    for source_dir in source_dirs:
        source = source_dir / config_name
        if source.exists():
            _copy_if_missing(source, destination)
            return


def _prepare_runtime_files(bundle_root: Path, runtime_root: Path) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "config").mkdir(parents=True, exist_ok=True)
    (runtime_root / "data").mkdir(parents=True, exist_ok=True)
    (runtime_root / "models").mkdir(parents=True, exist_ok=True)

    if not _is_frozen():
        return

    config_dirs = _config_source_dirs(bundle_root)
    for config_name in (
        "app_settings.json",
        "text_rules.json",
        "personal_dictionary.json",
        "llm_postedit_rules.json",
    ):
        _copy_config_if_missing(config_name=config_name, runtime_root=runtime_root, source_dirs=config_dirs)


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
    _ensure_standard_streams()
    _configure_hf_runtime_env()
    bundle_root = _bundle_root_dir()
    runtime_root = _runtime_root_dir(bundle_root)
    _prepare_runtime_files(bundle_root=bundle_root, runtime_root=runtime_root)

    settings_path = runtime_root / "config" / "app_settings.json"
    settings = load_json(settings_path)

    setup_logging(settings.get("log_level", "INFO"), runtime_root=runtime_root)

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
        "enabled": bool(settings.get("llm_enabled", False)),
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
        "external_agent_enabled": bool(settings.get("external_agent_enabled", False)),
        "external_agent_url": str(
            settings.get("external_agent_url", "http://127.0.0.1:8000/v1/agent/chat")
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
