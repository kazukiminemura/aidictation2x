"""LLM post-editor orchestrator.

Coordinates chunking, backend calls, and quality gating.
Backend implementations live in llm_backends.py (OCP: add backends without editing here).
"""
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .llm_backends import (
    LLMBackend,
    OllamaBackend,
    OpenVINOBackend,
    RuleFileBackend,
    _is_openvino_model_ref,
)
from .quality_gate import QualityGate
from .text_processing import create_edit_list


@dataclass
class LLMOptions:
    enabled: bool
    strength: str  # "weak" | "medium" | "strong"
    max_input_chars: int
    max_change_ratio: float
    domain_hint: str


@dataclass
class LLMResult:
    final_text: str
    applied: bool
    fallback_reason: str
    edits: list[str]
    latency_ms: int


class LLMPostEditor:
    def __init__(
        self,
        model_path: Path,
        timeout_ms: int = 8000,
        blocked_patterns: list[str] | None = None,
        backend: LLMBackend | None = None,
        llm_device: str = "GPU",
        auto_download: bool = True,
        download_dir: Path | None = None,
    ):
        self.model_path = model_path
        self.timeout_ms = timeout_ms
        self.llm_device = llm_device
        self.auto_download = auto_download
        self.download_dir = download_dir or Path("models") / "openvino"
        self.logger = logging.getLogger(__name__)
        self.quality_gate = QualityGate(blocked_patterns or [])
        self.backend = backend or self._resolve_backend(model_path)

    def download_model(self) -> str:
        if self.backend is None:
            raise RuntimeError("backend_unavailable")
        if not hasattr(self.backend, "download_model"):
            raise RuntimeError("download_not_supported_for_backend")
        return str(self.backend.download_model())

    def get_download_target_dir(self) -> Path | None:
        if self.backend is None:
            return None
        if hasattr(self.backend, "get_download_target_dir"):
            return self.backend.get_download_target_dir()
        return None

    def refine(self, raw_text: str, preprocessed_text: str, options: LLMOptions) -> LLMResult:  # noqa: ARG002
        started = time.perf_counter()

        if not options.enabled:
            return self._build_result(preprocessed_text, False, "disabled", [], started)

        if not preprocessed_text.strip():
            return self._build_result(preprocessed_text, False, "empty_input", [], started)

        if self.backend is None:
            return self._build_result(preprocessed_text, False, "backend_unavailable", [], started)

        try:
            chunks = self._chunk_text(preprocessed_text, max(100, options.max_input_chars))
            refined_chunks = [self.backend.generate(chunk, options, self.timeout_ms) for chunk in chunks]
            candidate = "".join(refined_chunks).strip()
        except subprocess.TimeoutExpired:
            return self._build_result(preprocessed_text, False, "timeout", [], started)
        except RuntimeError as exc:
            reason = str(exc).strip() or "llm_runtime_error"
            if reason in {
                "model_not_found_and_auto_download_disabled",
                "openvino_genai_not_installed",
                "huggingface_hub_not_installed",
                "model_download_failed",
                "download_not_supported_for_backend",
            }:
                self.logger.warning("LLM skipped: %s", reason)
                return self._build_result(preprocessed_text, False, reason, [], started)
            self.logger.exception("LLM runtime error")
            return self._build_result(preprocessed_text, False, "llm_error", [], started)
        except Exception:  # noqa: BLE001
            self.logger.exception("LLM refinement failed")
            return self._build_result(preprocessed_text, False, "llm_error", [], started)

        gate = self.quality_gate.validate(preprocessed_text, candidate, options.max_change_ratio)
        if not gate.accepted:
            return self._build_result(preprocessed_text, False, gate.reason, [], started)

        edits = create_edit_list(preprocessed_text, candidate)
        return self._build_result(candidate, True, "", edits, started)

    def _resolve_backend(self, model_path: Path) -> LLMBackend | None:
        import shutil as _shutil
        model_ref = str(model_path)

        if _is_openvino_model_ref(model_ref):
            return OpenVINOBackend(
                model_ref=model_ref,
                device=self.llm_device,
                auto_download=self.auto_download,
                download_dir=self.download_dir,
            )

        if model_path.is_file() and model_path.suffix.lower() == ".json":
            return RuleFileBackend(model_path)

        rules_file = model_path / "rules.json"
        if model_path.is_dir() and rules_file.exists():
            return RuleFileBackend(rules_file)

        if model_path.exists() and model_path.is_dir():
            return OpenVINOBackend(
                model_ref=model_ref,
                device=self.llm_device,
                auto_download=self.auto_download,
                download_dir=self.download_dir,
            )

        if _shutil.which("ollama"):
            return OllamaBackend(model_ref=model_ref)

        return None

    @staticmethod
    def _chunk_text(text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        parts = re.split(r"(?<=[。！？!?])", text)
        chunks: list[str] = []
        current = ""
        for part in parts:
            if not part:
                continue
            if len(current) + len(part) <= max_chars:
                current += part
                continue
            if current:
                chunks.append(current)
            if len(part) <= max_chars:
                current = part
                continue
            for idx in range(0, len(part), max_chars):
                chunks.append(part[idx : idx + max_chars])
            current = ""

        if current:
            chunks.append(current)
        return chunks or [text]

    def _build_result(
        self,
        text: str,
        applied: bool,
        fallback_reason: str,
        edits: list[str],
        started: float,
    ) -> LLMResult:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResult(
            final_text=text,
            applied=applied,
            fallback_reason=fallback_reason,
            edits=edits,
            latency_ms=latency_ms,
        )
