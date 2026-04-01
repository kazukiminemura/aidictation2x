import re
from pathlib import Path
from typing import Callable

import numpy as np

_OV_EXPORT_TASK = "automatic-speech-recognition"
_DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
_SUPPORTED_MODEL_IDS = (
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B",
    "openai/whisper-base",
    "openai/whisper-large-v3-turbo",
)


class _WhisperIREngine:
    def __init__(self, model_dir: str, device: str):
        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError("openvino_genai_not_installed") from exc

        ov_device = _to_openvino_device(device)
        self.pipeline = ov_genai.WhisperPipeline(model_dir, ov_device)
        self.generation_config = self.pipeline.get_generation_config()
        ja_key = _select_japanese_language_key(self.generation_config)
        if ja_key:
            self.generation_config.language = ja_key
        self.generation_config.task = "transcribe"
        self.generation_config.return_timestamps = False
        self.generation_config_auto = self.pipeline.get_generation_config()
        self.generation_config_auto.task = "transcribe"
        self.generation_config_auto.return_timestamps = False
        self.max_chunk_samples = 4 * 16000
        self.min_chunk_samples = max(1, int(0.25 * 16000))

    def transcribe(self, audio_data: np.ndarray) -> str:
        if audio_data.size == 0:
            return ""

        audio = np.asarray(audio_data, dtype=np.float32)
        texts = self._transcribe_with_config(audio, self.generation_config)
        if not texts and _has_voice(audio):
            texts = self._transcribe_with_config(audio, self.generation_config_auto)
        return _dedupe_repeated_text(" ".join(texts).strip())

    def _transcribe_with_config(self, audio: np.ndarray, config) -> list[str]:  # noqa: ANN001
        texts: list[str] = []
        for start in range(0, int(audio.size), self.max_chunk_samples):
            chunk = audio[start : start + self.max_chunk_samples]
            if chunk.size == 0:
                continue
            texts.extend(self._transcribe_chunk_recursive(chunk, config))
        return texts

    def _transcribe_chunk_recursive(self, chunk: np.ndarray, config) -> list[str]:  # noqa: ANN001
        try:
            result = self.pipeline.generate(np.asarray(chunk, dtype=np.float32).tolist(), config)
            chunk_texts = list(getattr(result, "texts", []))
            if not chunk_texts:
                return []
            text = str(chunk_texts[0]).strip()
            return [text] if text else []
        except Exception as exc:  # noqa: BLE001
            if "vector too long" not in str(exc).lower() or int(chunk.size) <= self.min_chunk_samples:
                raise
            mid = int(chunk.size // 2)
            if mid <= 0:
                raise
            texts: list[str] = []
            left = chunk[:mid]
            right = chunk[mid:]
            if left.size:
                texts.extend(self._transcribe_chunk_recursive(left, config))
            if right.size:
                texts.extend(self._transcribe_chunk_recursive(right, config))
            return texts


class _QwenASREngine:
    def __init__(self, model_source: str, device: str, language: str | None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch_not_installed") from exc
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError("qwen_asr_not_installed") from exc

        self.language = _normalize_language(language)
        self.device_map = _to_qwen_device_map(device, torch)
        self.model = Qwen3ASRModel.from_pretrained(
            model_source,
            dtype=torch.bfloat16 if self.device_map.startswith("cuda") else torch.float32,
            device_map=self.device_map,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio_data: np.ndarray, sample_rate_hz: int) -> str:
        if audio_data.size == 0:
            return ""
        results = self.model.transcribe(
            audio=(np.asarray(audio_data, dtype=np.float32), int(sample_rate_hz)),
            language=self.language,
        )
        if not results:
            return ""
        first = results[0]
        text = getattr(first, "text", first)
        return _dedupe_repeated_text(str(text).strip())


class ASREngine:
    def __init__(
        self,
        sample_rate_hz: int,
        device: str = "gpu",
        model_id: str = _DEFAULT_MODEL_ID,
        models_root_dir: Path | None = None,
        language: str = "ja",
    ):
        self.sample_rate_hz = sample_rate_hz
        self.device = device
        self.model_id = _normalize_model_id(model_id)
        self.models_root_dir = models_root_dir or Path("models") / "asr"
        self.language = language
        self._engine: _WhisperIREngine | _QwenASREngine | None = None

    def configure(self, device: str | None = None, model_id: str | None = None, language: str | None = None) -> None:
        if device is not None and device != self.device:
            self.device = device
            self._engine = None
        if model_id is not None:
            normalized_model_id = _normalize_model_id(model_id)
            if normalized_model_id != self.model_id:
                self.model_id = normalized_model_id
                self._engine = None
        if language is not None and language != self.language:
            self.language = language
            self._engine = None

    def transcribe(self, audio_data: np.ndarray) -> str:
        audio = np.asarray(audio_data, dtype=np.float32)
        if audio.size == 0 or not _has_voice(audio):
            return ""

        if self._engine is None:
            self._engine = self._build_engine()
        if _is_qwen_model(self.model_id):
            assert isinstance(self._engine, _QwenASREngine)
            return self._engine.transcribe(audio, self.sample_rate_hz).strip()
        assert isinstance(self._engine, _WhisperIREngine)
        return self._engine.transcribe(audio).strip()

    def _build_engine(self) -> _WhisperIREngine | _QwenASREngine:
        if _is_qwen_model(self.model_id):
            if not _looks_like_qwen_model_dir(self.get_model_dir()):
                self.convert_model()
            return _QwenASREngine(
                model_source=str(self.get_model_dir()),
                device=self.device,
                language=self.language,
            )
        if not _looks_like_openvino_model_dir(self.get_model_dir()):
            self.convert_model()
        return _WhisperIREngine(str(self.get_model_dir()), self.device)

    def convert_model(self, progress_callback: Callable[[str], None] | None = None) -> str:
        if _is_qwen_model(self.model_id):
            try:
                from huggingface_hub import snapshot_download
                from huggingface_hub.utils import disable_progress_bars
            except ImportError as exc:
                raise RuntimeError("huggingface_hub_not_installed") from exc

            model_dir = self.get_model_dir()
            model_dir.mkdir(parents=True, exist_ok=True)
            _report_progress(progress_callback, f"Downloading ASR files for {self.get_display_name()}...")
            disable_progress_bars()
            kwargs = {
                "repo_id": self.model_id,
                "local_dir": str(model_dir),
                "local_dir_use_symlinks": False,
            }
            try:
                snapshot_download(**kwargs)
            except TypeError:
                kwargs.pop("local_dir_use_symlinks", None)
                snapshot_download(**kwargs)
            _report_progress(progress_callback, f"Validating downloaded files for {self.get_display_name()}...")
            if not _looks_like_qwen_model_dir(model_dir):
                raise RuntimeError("model_download_failed")
            self._engine = None
            return str(model_dir)

        try:
            from optimum.exporters.openvino.convert import export_tokenizer
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("openvino_export_dependencies_not_installed") from exc

        model_dir = self.get_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        _report_progress(progress_callback, f"Downloading processor for {self.get_display_name()}...")
        processor = AutoProcessor.from_pretrained(self.model_id)
        _report_progress(progress_callback, f"Exporting {self.get_display_name()} to OpenVINO IR...")
        model = OVModelForSpeechSeq2Seq.from_pretrained(self.model_id, export=True, compile=False)
        _report_progress(progress_callback, f"Saving processor files for {self.get_display_name()}...")
        processor.save_pretrained(model_dir)
        _report_progress(progress_callback, f"Saving OpenVINO IR files for {self.get_display_name()}...")
        model.save_pretrained(model_dir)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            _report_progress(progress_callback, f"Exporting tokenizer for {self.get_display_name()}...")
            export_tokenizer(tokenizer, model_dir, task=_OV_EXPORT_TASK)
        _report_progress(progress_callback, f"Validating exported files for {self.get_display_name()}...")
        if not _looks_like_openvino_model_dir(model_dir):
            raise RuntimeError("model_download_failed")
        self._engine = None
        return str(model_dir)

    def get_model_dir(self) -> Path:
        return self.models_root_dir / _model_dir_name(self.model_id)

    def get_display_name(self) -> str:
        return self.model_id.split("/", 1)[-1]


def _is_qwen_model(model_id: str) -> bool:
    return model_id.startswith("Qwen/")


def _to_openvino_device(device: str) -> str:
    normalized = (device or "gpu").strip().lower()
    if normalized == "cpu":
        return "CPU"
    if normalized in {"gpu", "cuda"}:
        return "GPU"
    if normalized == "npu":
        return "NPU"
    return "GPU"


def _to_qwen_device_map(device: str, torch_module) -> str:  # noqa: ANN001
    normalized = (device or "gpu").strip().lower()
    if normalized in {"cpu", "npu"}:
        return "cpu"
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _normalize_language(language: str | None) -> str | None:
    normalized = (language or "").strip()
    if not normalized:
        return None
    aliases = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
    return aliases.get(normalized.lower(), normalized)


def _looks_like_qwen_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    required = ("config.json", "tokenizer_config.json")
    if not all((path / name).exists() for name in required):
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


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


def _has_voice(audio_data: np.ndarray) -> bool:
    if audio_data.size == 0:
        return False
    return float(np.mean(np.abs(np.asarray(audio_data, dtype=np.float32)))) >= 0.003


def get_supported_model_ids() -> tuple[str, ...]:
    return _SUPPORTED_MODEL_IDS


def _normalize_model_id(model_id: str) -> str:
    normalized = (model_id or "").strip() or _DEFAULT_MODEL_ID
    if normalized not in _SUPPORTED_MODEL_IDS:
        raise RuntimeError(f"unsupported_asr_model: {normalized}")
    return normalized


def _model_dir_name(model_id: str) -> str:
    return model_id.split("/", 1)[-1].replace("/", "--")


def _report_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _dedupe_repeated_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    return _dedupe_exact_repeat(_dedupe_adjacent_sentence_runs(cleaned))


def _dedupe_adjacent_sentence_runs(text: str) -> str:
    parts = [part.strip() for part in re.split(r"([.!?\n]+)", text) if part]
    if not parts:
        return text
    merged: list[str] = []
    last_normalized = ""
    i = 0
    while i < len(parts):
        sentence = parts[i].strip()
        suffix = parts[i + 1] if i + 1 < len(parts) and re.fullmatch(r"[.!?\n]+", parts[i + 1]) else ""
        chunk = f"{sentence}{suffix}".strip()
        normalized = _normalize_repeat_key(chunk)
        if chunk and normalized != last_normalized:
            merged.append(chunk)
            last_normalized = normalized
        i += 2 if suffix else 1
    return " ".join(part for part in merged if part).strip()


def _dedupe_exact_repeat(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    for unit_len in range(1, len(compact) // 2 + 1):
        if len(compact) % unit_len != 0:
            continue
        unit = compact[:unit_len].strip()
        if not unit:
            continue
        repeats = len(compact) // unit_len
        if repeats >= 2 and unit * repeats == compact:
            return unit
    return compact


def _normalize_repeat_key(text: str) -> str:
    return re.sub(r"[\s.!?\n]+", "", text).strip().lower()
