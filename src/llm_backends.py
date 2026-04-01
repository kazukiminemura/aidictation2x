"""LLM backend implementations (OpenVINO, Ollama, RuleFile).

Single Responsibility: Each class handles one inference backend.
Open/Closed: Add new backends by implementing LLMBackend protocol.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol


class LLMBackend(Protocol):
    def generate(self, text: str, options: Any, timeout_ms: int) -> str:
        ...


class OpenVINOBackend:
    """Local OpenVINO GenAI backend with optional auto-download from Hugging Face."""

    def __init__(
        self,
        model_ref: str,
        device: str = "CPU",
        auto_download: bool = True,
        download_dir: Path | None = None,
    ):
        self.model_ref = model_ref
        self.device = device
        self.auto_download = auto_download
        self.download_dir = download_dir or Path("models") / "openvino"
        self._pipeline: Any | None = None
        self._genai: Any | None = None
        self._resolved_model_path: str | None = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError("openvino_genai_not_installed") from exc

        self._genai = ov_genai
        model_source = self._resolve_model_source()
        self._pipeline = ov_genai.LLMPipeline(model_source, self.device)

    def _resolve_model_source(self, force_download: bool = False) -> str:
        if self._resolved_model_path is not None:
            return self._resolved_model_path

        model_path = Path(self.model_ref)
        if model_path.exists():
            self._resolved_model_path = str(model_path)
            return self._resolved_model_path

        repo_id = _normalize_repo_id(self.model_ref)

        if not _is_openvino_model_ref(repo_id):
            self._resolved_model_path = self.model_ref
            return self._resolved_model_path

        self.download_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.download_dir / repo_id.replace("/", "--")
        if _looks_like_model_dir(target_dir):
            self._resolved_model_path = str(target_dir)
            return self._resolved_model_path

        if not self.auto_download and not force_download:
            raise RuntimeError("model_not_found_and_auto_download_disabled")

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars
        except ImportError as exc:
            raise RuntimeError("huggingface_hub_not_installed") from exc

        disable_progress_bars()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        kwargs = {
            "repo_id": repo_id,
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,
        }
        try:
            snapshot_download(tqdm_class=None, **kwargs)
        except TypeError:
            snapshot_download(**kwargs)

        if not _looks_like_model_dir(target_dir):
            raise RuntimeError("model_download_failed")

        self._resolved_model_path = str(target_dir)
        return self._resolved_model_path

    def download_model(self) -> str:
        return self._resolve_model_source(force_download=True)

    def get_download_target_dir(self) -> Path | None:
        model_path = Path(self.model_ref)
        if model_path.exists():
            return model_path if model_path.is_dir() else model_path.parent

        repo_id = _normalize_repo_id(self.model_ref)
        if not _is_openvino_model_ref(repo_id):
            return None
        return self.download_dir / repo_id.replace("/", "--")

    def generate(self, text: str, options: Any, timeout_ms: int) -> str:
        self._ensure_pipeline()
        assert self._pipeline is not None

        prompt = _build_prompt(text=text, options=options)
        max_new_tokens = _estimate_max_new_tokens(text)

        output_text: str | None = None
        try:
            gen_cfg = self._genai.GenerationConfig()
            gen_cfg.max_new_tokens = max_new_tokens
            gen_cfg.temperature = 0.0
            gen_cfg.top_p = 1.0
            gen_cfg.do_sample = False
            output = self._pipeline.generate(prompt, gen_cfg)
            output_text = _coerce_generation_output(output)
        except Exception:  # noqa: BLE001
            output = self._pipeline.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            output_text = _coerce_generation_output(output)

        if not output_text:
            raise RuntimeError("empty_output")

        return _post_process_model_output(output_text, prompt)


class OllamaBackend:
    def __init__(self, model_ref: str):
        self.model_ref = model_ref

    def generate(self, text: str, options: Any, timeout_ms: int) -> str:
        prompt = _build_prompt(text=text, options=options)
        result = subprocess.run(
            ["ollama", "run", self.model_ref],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=max(1, timeout_ms // 1000),
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or "ollama_failed")

        output = (result.stdout or "").strip()
        if not output:
            raise RuntimeError("empty_output")

        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return lines[-1] if lines else output


class RuleFileBackend:
    def __init__(self, rules_file: Path):
        with rules_file.open("r", encoding="utf-8-sig") as fp:
            payload = json.load(fp)
        self.replacements = payload.get("replacements", {})

    def generate(self, text: str, options: Any, timeout_ms: int) -> str:
        _ = timeout_ms
        output = text
        for source, target in self.replacements.items():
            output = output.replace(source, target)

        if options.strength in {"medium", "strong"}:
            output = _normalize_for_medium(output)
        if options.strength == "strong":
            output = _normalize_for_strong(output)

        return output


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_openvino_model_ref(model_ref: str) -> bool:
    normalized = _normalize_repo_id(model_ref)
    return normalized.startswith("OpenVINO/")


def _normalize_repo_id(model_ref: str) -> str:
    return model_ref.strip().strip("'\"").replace("\\", "/")


def _coerce_generation_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()

    if hasattr(output, "texts"):
        texts = getattr(output, "texts")
        if isinstance(texts, list) and texts:
            return str(texts[0]).strip()

    return str(output).strip()


def _build_prompt(text: str, options: Any) -> str:
    return (
        "あなたは日本語の音声認識後編集器です。\n"
        "意味を変えず、誤変換だけを修正してください。要約・創作・言い換えは禁止です。\n"
        f"補正強度: {options.strength}\n"
        f"専門用語ヒント: {options.domain_hint or 'なし'}\n"
        "出力は修正後テキストのみを1つ返してください。\n"
        "--- 入力 ---\n"
        f"{text}"
    )


def _post_process_model_output(output: str, prompt: str) -> str:
    text = output.replace(prompt, "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _estimate_max_new_tokens(text: str) -> int:
    return max(64, min(512, int(len(text) * 0.8)))


def _normalize_for_medium(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in "。．.!！？?":
        text += "。"
    return text


def _normalize_for_strong(text: str) -> str:
    text = text.replace("  ", " ")
    text = text.replace(" ,", "、")
    return text


def _looks_like_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    xml_files = list(path.rglob("*.xml"))
    if not xml_files:
        return False
    for xml_file in xml_files:
        if xml_file.with_suffix(".bin").exists():
            return True
    return False
