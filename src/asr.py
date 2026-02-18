import os
from pathlib import Path
from typing import Any

import numpy as np

_WHISPER_MODEL_REPOS = {
    "tiny": "OpenVINO/whisper-tiny",
    "base": "OpenVINO/whisper-base",
    "small": "OpenVINO/whisper-small",
    "medium": "OpenVINO/whisper-medium",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo",
}


class _WhisperEngine:
    def __init__(self, model_name: str, device: str, compute_type: str):  # noqa: ARG002
        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError("openvino_genai_not_installed") from exc

        ov_device = _to_openvino_device(device)
        self.pipeline = ov_genai.WhisperPipeline(model_name, ov_device)
        # Use model-provided defaults; constructing empty config can miss required maps (e.g. lang_to_id).
        self.generation_config = self.pipeline.get_generation_config()
        ja_key = _select_japanese_language_key(self.generation_config)
        if ja_key:
            self.generation_config.language = ja_key
        self.generation_config.task = "transcribe"
        self.generation_config.return_timestamps = False
        self.generation_config_auto = self.pipeline.get_generation_config()
        self.generation_config_auto.task = "transcribe"
        self.generation_config_auto.return_timestamps = False
        # Keep chunks conservative and split again on runtime "vector too long" errors.
        self.max_chunk_samples = 4 * 16000
        self.min_chunk_samples = max(1, int(0.25 * 16000))

    def transcribe(self, audio_data: np.ndarray) -> str:
        if audio_data.size == 0:
            return ""

        audio = np.asarray(audio_data, dtype=np.float32)
        texts = self._transcribe_with_config(audio, self.generation_config)
        if not texts and _has_voice(audio):
            # Fallback to auto language detection when explicit Japanese key yields empty output.
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


class _QwenASREngine:
    def __init__(self, model_name: str, device: str, compute_type: str, sample_rate_hz: int):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch_not_installed") from exc
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError("qwen_asr_not_installed") from exc

        self.sample_rate_hz = sample_rate_hz
        self._torch = torch
        dtype = self._resolve_dtype(device=device, compute_type=compute_type)
        device_map = self._resolve_device_map(device=device)
        self.model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio_data: np.ndarray) -> str:
        audio = np.asarray(audio_data, dtype=np.float32)
        if audio.size == 0:
            return ""

        results = self.model.transcribe(
            audio=(audio, self.sample_rate_hz),
            language=None,
        )
        if not results:
            return ""

        first = results[0]
        text = getattr(first, "text", "")
        return str(text).strip()

    def _resolve_dtype(self, device: str, compute_type: str):  # noqa: ANN001
        normalized_device = (device or "auto").strip().lower()
        normalized_compute = (compute_type or "").strip().lower()
        if normalized_device == "cpu":
            return self._torch.float32
        if normalized_compute == "float32":
            return self._torch.float32
        if normalized_compute in {"float16", "int8_float16"}:
            return self._torch.float16
        return self._torch.bfloat16

    def _resolve_device_map(self, device: str) -> str:
        normalized_device = (device or "auto").strip().lower()
        if normalized_device == "cpu":
            return "cpu"
        if normalized_device in {"cuda", "gpu"}:
            return "cuda:0"
        if self._torch.cuda.is_available():
            return "cuda:0"
        return "cpu"


class ASREngine:
    def __init__(
        self,
        sample_rate_hz: int,
        whisper_model_name: str = "Qwen/Qwen3-ASR-0.6B",
        whisper_device: str = "auto",
        whisper_compute_type: str = "int8",
        whisper_download_dir: Path | None = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.whisper_model_name = whisper_model_name
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.whisper_download_dir = whisper_download_dir or Path("models") / "whisper"
        self._engine: Any | None = None

    def configure(
        self,
        whisper_model_name: str | None = None,
        whisper_device: str | None = None,
        whisper_compute_type: str | None = None,
        whisper_download_dir: Path | None = None,
    ) -> None:
        changed = False

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
            # Last-resort fallback: split by short fixed windows and skip only failing windows.
            window = max(1, int(self.sample_rate_hz * 1))
            texts: list[str] = []
            failed_windows = 0
            for start in range(0, int(audio.size), window):
                chunk = audio[start : start + window]
                if chunk.size == 0:
                    continue
                try:
                    text = self._engine.transcribe(chunk).strip()
                except Exception as inner_exc:  # noqa: BLE001
                    if "vector too long" in str(inner_exc).lower():
                        failed_windows += 1
                        continue
                    raise
                if text:
                    texts.append(text)
            joined = " ".join(texts).strip()
            if joined:
                return joined
            if _has_voice(audio):
                if failed_windows > 0:
                    raise RuntimeError("asr_failed_all_windows")
                raise RuntimeError("asr_empty_output")
            return ""
        if text:
            return text
        audio = np.asarray(audio_data, dtype=np.float32)
        if _has_voice(audio):
            raise RuntimeError("asr_empty_output")
        return ""

    def _build_engine(self) -> Any:
        backend = _resolve_asr_backend(self.whisper_model_name)
        model_source = self._resolve_model_source(backend=backend)
        if backend == "qwen":
            return _QwenASREngine(
                model_name=model_source,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
                sample_rate_hz=self.sample_rate_hz,
            )
        return _WhisperEngine(
            model_name=model_source,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
        )

    def download_whisper_model(self, model_name: str | None = None) -> str:
        target_model = (model_name or self.whisper_model_name).strip()
        if not target_model:
            raise RuntimeError("whisper_model_name_missing")

        local_path = Path(target_model)
        if local_path.exists():
            return str(local_path)

        backend = _resolve_asr_backend(target_model)
        repo_id = _resolve_whisper_repo_id(target_model)
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars
        except ImportError as exc:
            raise RuntimeError("huggingface_hub_not_installed") from exc

        disable_progress_bars()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self.whisper_download_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.whisper_download_dir / repo_id.replace("/", "--")
        kwargs = {
            "repo_id": repo_id,
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,
        }
        try:
            snapshot_download(tqdm_class=None, **kwargs)
        except TypeError:
            snapshot_download(**kwargs)
        if backend == "whisper" and not _looks_like_openvino_model_dir(target_dir):
            raise RuntimeError("whisper_model_download_failed")
        self.whisper_model_name = str(target_dir)
        self._engine = None
        return str(target_dir)

    def get_whisper_download_target_dir(self, model_name: str | None = None) -> Path | None:
        target_model = (model_name or self.whisper_model_name).strip()
        if not target_model:
            return None

        local_path = Path(target_model)
        if local_path.exists():
            return local_path if local_path.is_dir() else local_path.parent

        repo_id = _resolve_whisper_repo_id(target_model)
        return self.whisper_download_dir / repo_id.replace("/", "--")

    def _resolve_model_source(self, backend: str) -> str:
        model_name = (self.whisper_model_name or "").strip()
        if not model_name:
            raise RuntimeError("whisper_model_name_missing")

        local_model_dir = Path(model_name)
        if local_model_dir.exists():
            if backend == "qwen":
                return str(local_model_dir)
            if _looks_like_openvino_model_dir(local_model_dir):
                return str(local_model_dir)

        repo_id = _resolve_whisper_repo_id(model_name)
        cached_model_dir = self.whisper_download_dir / repo_id.replace("/", "--")
        if cached_model_dir.exists():
            if backend == "qwen":
                return str(cached_model_dir)
            if _looks_like_openvino_model_dir(cached_model_dir):
                return str(cached_model_dir)

        if backend == "qwen":
            return repo_id

        # Fallback: auto-download when no local model is available.
        try:
            downloaded_path = self.download_whisper_model(model_name=model_name)
            if _looks_like_openvino_model_dir(Path(downloaded_path)):
                return downloaded_path
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"whisper_model_download_failed: {exc}") from exc

        raise RuntimeError(
            "whisper_model_not_found. Please pre-download from Properties -> Download ASR Model."
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


def _resolve_asr_backend(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if "qwen3-asr" in normalized or normalized.startswith("qwen/"):
        return "qwen"
    return "whisper"


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


def _has_voice(audio_data: np.ndarray) -> bool:
    if audio_data.size == 0:
        return False
    audio = np.asarray(audio_data, dtype=np.float32)
    return float(np.mean(np.abs(audio))) >= 0.003
