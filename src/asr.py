from pathlib import Path

import numpy as np

_MODEL_DIR_NAME = "whisper-large-v3-turbo"
_HF_MODEL_ID = "openai/whisper-large-v3-turbo"
_OV_EXPORT_TASK = "automatic-speech-recognition"


class _WhisperEngine:
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
        return " ".join(texts).strip()

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
            result = self.pipeline.generate(
                np.asarray(chunk, dtype=np.float32).tolist(),
                config,
            )
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
            left = chunk[:mid]
            right = chunk[mid:]
            texts: list[str] = []
            if left.size:
                texts.extend(self._transcribe_chunk_recursive(left, config))
            if right.size:
                texts.extend(self._transcribe_chunk_recursive(right, config))
            return texts


class ASREngine:
    def __init__(
        self,
        sample_rate_hz: int,
        device: str = "auto",
        model_dir: Path | None = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.device = device
        self.model_dir = model_dir or Path("models") / "whisper" / _MODEL_DIR_NAME
        self._engine: _WhisperEngine | None = None

    def configure(self, device: str | None = None) -> None:
        if device is not None and device != self.device:
            self.device = device
            self._engine = None

    def transcribe(self, audio_data: np.ndarray) -> str:
        audio = np.asarray(audio_data, dtype=np.float32)
        if audio.size == 0 or not _has_voice(audio):
            return ""

        if self._engine is None:
            self._engine = self._build_engine()
        try:
            text = self._engine.transcribe(audio).strip()
        except Exception as exc:  # noqa: BLE001
            if "vector too long" not in str(exc).lower():
                raise
            window = max(1, int(self.sample_rate_hz * 1))
            texts: list[str] = []
            for start in range(0, int(audio.size), window):
                chunk = audio[start : start + window]
                if chunk.size == 0:
                    continue
                try:
                    text = self._engine.transcribe(chunk).strip()
                except Exception as inner_exc:  # noqa: BLE001
                    if "vector too long" in str(inner_exc).lower():
                        continue
                    raise
                if text:
                    texts.append(text)
            return " ".join(texts).strip()
        return text

    def _build_engine(self) -> _WhisperEngine:
        if not _looks_like_openvino_model_dir(self.model_dir):
            self.convert_model()
        return _WhisperEngine(str(self.model_dir), self.device)

    def convert_model(self) -> str:
        """Download and convert openai/whisper-large-v3-turbo into OpenVINO IR."""
        try:
            from optimum.exporters.openvino.convert import export_tokenizer
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("openvino_export_dependencies_not_installed") from exc

        self.model_dir.mkdir(parents=True, exist_ok=True)
        processor = AutoProcessor.from_pretrained(_HF_MODEL_ID)
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            _HF_MODEL_ID,
            export=True,
            compile=False,
        )
        processor.save_pretrained(self.model_dir)
        model.save_pretrained(self.model_dir)

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            export_tokenizer(tokenizer, self.model_dir, task=_OV_EXPORT_TASK)

        if not _looks_like_openvino_model_dir(self.model_dir):
            raise RuntimeError("model_download_failed")
        self._engine = None
        return str(self.model_dir)

    def get_model_dir(self) -> Path:
        return self.model_dir


def _to_openvino_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized == "cpu":
        return "CPU"
    if normalized in {"gpu", "cuda"}:
        return "GPU"
    if normalized == "npu":
        return "NPU"
    # auto: try NPU first, then GPU, then CPU
    return "AUTO:NPU,GPU,CPU"


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
    audio = np.asarray(audio_data, dtype=np.float32)
    return float(np.mean(np.abs(audio))) >= 0.003
