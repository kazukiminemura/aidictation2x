import json
from pathlib import Path

import numpy as np
from vosk import KaldiRecognizer, Model

_WHISPER_MODEL_REPOS = {
    "tiny": "OpenVINO/whisper-tiny",
    "base": "OpenVINO/whisper-base",
    "small": "OpenVINO/whisper-small",
    "medium": "OpenVINO/whisper-medium",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo",
}


class _VoskEngine:
    def __init__(self, model_dir: Path, sample_rate_hz: int):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Vosk model not found: {model_dir}. Please place the model under models/."
            )
        self.model = Model(str(model_dir))
        self.sample_rate_hz = sample_rate_hz

    def transcribe(self, audio_data: np.ndarray) -> str:
        if audio_data.size == 0:
            return ""

        pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        rec = KaldiRecognizer(self.model, self.sample_rate_hz)
        rec.SetWords(True)
        rec.AcceptWaveform(pcm)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()


class _WhisperEngine:
    def __init__(self, model_name: str, device: str, compute_type: str):  # noqa: ARG002
        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError("openvino_genai_not_installed") from exc

        ov_device = _to_openvino_device(device)
        self.pipeline = ov_genai.WhisperPipeline(model_name, ov_device)
        self.generation_config = ov_genai.WhisperGenerationConfig()
        self.generation_config.language = _select_japanese_language_key(self.generation_config)
        self.generation_config.task = "transcribe"
        self.generation_config.return_timestamps = False

    def transcribe(self, audio_data: np.ndarray) -> str:
        if audio_data.size == 0:
            return ""

        result = self.pipeline.generate(
            np.asarray(audio_data, dtype=np.float32).tolist(),
            self.generation_config,
        )
        texts = list(getattr(result, "texts", []))
        if texts:
            return str(texts[0]).strip()
        return ""


class ASREngine:
    def __init__(
        self,
        sample_rate_hz: int,
        backend: str = "vosk",
        vosk_model_dir: Path | None = None,
        whisper_model_name: str = "OpenVINO/whisper-large-v3-int8-ov",
        whisper_device: str = "auto",
        whisper_compute_type: str = "int8",
        whisper_download_dir: Path | None = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.backend = backend
        self.vosk_model_dir = vosk_model_dir or Path("models") / "vosk-model-ja"
        self.whisper_model_name = whisper_model_name
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.whisper_download_dir = whisper_download_dir or Path("models") / "whisper"
        self._engine: _VoskEngine | _WhisperEngine | None = None
        if self.backend.strip().lower() == "vosk":
            self._engine = self._build_engine()

    def configure(
        self,
        backend: str | None = None,
        vosk_model_dir: Path | None = None,
        whisper_model_name: str | None = None,
        whisper_device: str | None = None,
        whisper_compute_type: str | None = None,
        whisper_download_dir: Path | None = None,
    ) -> None:
        changed = False

        if backend is not None and backend != self.backend:
            self.backend = backend
            changed = True
        if vosk_model_dir is not None and vosk_model_dir != self.vosk_model_dir:
            self.vosk_model_dir = vosk_model_dir
            changed = True
        if whisper_model_name is not None and whisper_model_name != self.whisper_model_name:
            self.whisper_model_name = whisper_model_name
            changed = True
        if whisper_device is not None and whisper_device != self.whisper_device:
            self.whisper_device = whisper_device
            changed = True
        if whisper_compute_type is not None and whisper_compute_type != self.whisper_compute_type:
            self.whisper_compute_type = whisper_compute_type
            changed = True
        if whisper_download_dir is not None and whisper_download_dir != self.whisper_download_dir:
            self.whisper_download_dir = whisper_download_dir
            changed = True

        if changed:
            self._engine = None

    def transcribe(self, audio_data: np.ndarray) -> str:
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine.transcribe(audio_data)

    def _build_engine(self) -> _VoskEngine | _WhisperEngine:
        backend = (self.backend or "vosk").strip().lower()
        if backend == "whisper":
            model_source = self._resolve_whisper_model_source()
            return _WhisperEngine(
                model_name=model_source,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        return _VoskEngine(model_dir=self.vosk_model_dir, sample_rate_hz=self.sample_rate_hz)

    def download_whisper_model(self, model_name: str | None = None) -> str:
        target_model = (model_name or self.whisper_model_name).strip()
        if not target_model:
            raise RuntimeError("whisper_model_name_missing")

        local_path = Path(target_model)
        if local_path.exists():
            return str(local_path)

        repo_id = _resolve_whisper_repo_id(target_model)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub_not_installed") from exc

        self.whisper_download_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.whisper_download_dir / repo_id.replace("/", "--")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        if not _looks_like_openvino_model_dir(target_dir):
            raise RuntimeError("whisper_model_download_failed")
        self.whisper_model_name = str(target_dir)
        self._engine = None
        return str(target_dir)

    def _resolve_whisper_model_source(self) -> str:
        model_name = (self.whisper_model_name or "").strip()
        if not model_name:
            raise RuntimeError("whisper_model_name_missing")

        local_model_dir = Path(model_name)
        if _looks_like_openvino_model_dir(local_model_dir):
            return str(local_model_dir)

        repo_id = _resolve_whisper_repo_id(model_name)
        cached_model_dir = self.whisper_download_dir / repo_id.replace("/", "--")
        if _looks_like_openvino_model_dir(cached_model_dir):
            return str(cached_model_dir)

        # Fallback: auto-download when no local model is available.
        try:
            downloaded_path = self.download_whisper_model(model_name=model_name)
            if _looks_like_openvino_model_dir(Path(downloaded_path)):
                return downloaded_path
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"whisper_model_download_failed: {exc}") from exc

        raise RuntimeError(
            "whisper_model_not_found. Please pre-download from Properties -> Download ASR Model (Whisper)."
        )


def _resolve_whisper_repo_id(model_name: str) -> str:
    model = model_name.strip()
    if not model:
        raise RuntimeError("whisper_model_name_missing")

    if "/" in model:
        return model

    if model in _WHISPER_MODEL_REPOS:
        return _WHISPER_MODEL_REPOS[model]

    return f"OpenVINO/whisper-{model}"


def _to_openvino_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized == "cpu":
        return "CPU"
    if normalized in {"gpu", "cuda"}:
        return "GPU"
    return "AUTO"


def _looks_like_openvino_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    xml_files = list(path.rglob("*.xml"))
    if not xml_files:
        return False
    return any(xml_file.with_suffix(".bin").exists() for xml_file in xml_files)


def _select_japanese_language_key(generation_config) -> str | None:  # noqa: ANN001
    lang_to_id = getattr(generation_config, "lang_to_id", {}) or {}
    if not isinstance(lang_to_id, dict) or not lang_to_id:
        return None

    for candidate in ("<|ja|>", "ja", "japanese"):
        if candidate in lang_to_id:
            return candidate

    for key in lang_to_id:
        normalized = str(key).strip().lower()
        if normalized in {"ja", "<|ja|>", "japanese"}:
            return str(key)

    return None
