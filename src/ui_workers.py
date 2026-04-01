"""WorkersMixin: model download workers and voice threshold calibration.

Mixed into VoiceInputApp. Accesses self.* attributes set in VoiceInputApp.__init__.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
import sounddevice as sd

from .ui_utils import parse_device_choice


class WorkersMixin:
    # ------------------------------------------------------------------
    # ASR model download
    # ------------------------------------------------------------------

    def _download_asr_model_clicked(self, model_id: str, device: str) -> None:
        if self._asr_download_in_progress:
            return
        self._start_asr_download_progress(model_id, device)
        self.status_var.set(f"Preparing ASR model: {model_id}")
        threading.Thread(
            target=self._download_asr_model_worker,
            args=(model_id, device),
            daemon=True,
        ).start()

    def _download_asr_model_worker(self, model_id: str, device: str) -> None:
        result: dict[str, str] = {"model_path": "", "error": ""}
        progress: dict[str, str] = {"phase": "Preparing..."}
        try:
            self.asr_engine.configure(device=device, model_id=model_id)
            target_dir = self.asr_engine.get_model_dir()

            def on_progress(message: str) -> None:
                progress["phase"] = message

            def run_convert() -> None:
                try:
                    result["model_path"] = self.asr_engine.convert_model(progress_callback=on_progress)
                except Exception as exc:  # noqa: BLE001
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        self.logger.exception("ASR model download failed")
                    except Exception:
                        pass

            convert_thread = threading.Thread(target=run_convert, daemon=True)
            convert_thread.start()
            started = time.perf_counter()
            while convert_thread.is_alive():
                elapsed_s = int(time.perf_counter() - started)
                downloaded = self._directory_size_bytes(target_dir)
                self.root.after(
                    0,
                    self._update_asr_download_progress,
                    (
                        f"{progress['phase']} "
                        f"[{self.asr_engine.get_display_name()} | {device.upper()}] "
                        f"{self._format_size(downloaded)} written "
                        f"({self._format_elapsed(elapsed_s)})"
                    ),
                )
                time.sleep(1.0)
            convert_thread.join()
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("ASR model conversion failed")
            result["error"] = str(exc)
        self.root.after(0, self._on_download_asr_model_done, result["model_path"], result["error"])

    def _on_download_asr_model_done(self, model_path: str, error: str) -> None:
        if error:
            self._stop_asr_download_progress("ASR model download failed")
            self.status_var.set("ASR model download failed")
            messagebox.showerror(
                "ASR model download error",
                f"{self._format_download_error(error)}\n\nLog: {self.root_dir / 'logs' / 'app.log'}",
            )
            return
        self._stop_asr_download_progress(f"ASR model ready: {Path(model_path).name}")
        self.status_var.set(f"ASR model ready: {Path(model_path).name}")
        messagebox.showinfo("ASR model", f"Model is ready at:\n{model_path}")

    def _start_asr_download_progress(self, model_id: str, device: str) -> None:
        self._asr_download_in_progress = True
        self.asr_download_progress_var.set(f"Preparing {model_id} on {device.upper()}...")
        if self.asr_download_button is not None:
            self.asr_download_button.config(state=tk.DISABLED)
        if self.asr_download_progressbar is not None:
            self.asr_download_progressbar.stop()
            self.asr_download_progressbar.config(mode="indeterminate")
            self.asr_download_progressbar.start(12)

    def _update_asr_download_progress(self, message: str) -> None:
        self.asr_download_progress_var.set(message)
        self.status_var.set(message)

    def _stop_asr_download_progress(self, message: str) -> None:
        self._asr_download_in_progress = False
        self.asr_download_progress_var.set(message)
        if self.asr_download_button is not None:
            self.asr_download_button.config(state=tk.NORMAL)
        if self.asr_download_progressbar is not None:
            self.asr_download_progressbar.stop()

    # ------------------------------------------------------------------
    # LLM model download
    # ------------------------------------------------------------------

    def _download_model_clicked(self) -> None:
        self.status_var.set("Downloading LLM model...")
        threading.Thread(target=self._download_model_worker, daemon=True).start()

    def _download_model_worker(self) -> None:
        result: dict[str, str] = {"model_path": "", "error": ""}

        def run_download() -> None:
            try:
                result["model_path"] = self.llm_editor.download_model()
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"
                try:
                    self.logger.exception("Model download failed")
                except Exception:
                    pass

        target_dir = self.llm_editor.get_download_target_dir()
        download_thread = threading.Thread(target=run_download, daemon=True)
        download_thread.start()
        started = time.perf_counter()
        while download_thread.is_alive():
            elapsed_s = int(time.perf_counter() - started)
            downloaded = self._directory_size_bytes(target_dir)
            self.root.after(
                0,
                self.status_var.set,
                (
                    "Downloading LLM model... "
                    f"{self._format_size(downloaded)} downloaded "
                    f"({self._format_elapsed(elapsed_s)})"
                ),
            )
            time.sleep(1.0)
        download_thread.join()
        self.root.after(0, self._on_download_model_done, result["model_path"], result["error"])

    def _on_download_model_done(self, model_path: str, error: str) -> None:
        if error:
            self.status_var.set("Model download failed")
            messagebox.showerror(
                "LLM model download error",
                f"{self._format_download_error(error)}\n\nLog: {self.root_dir / 'logs' / 'app.log'}",
            )
            return
        self.status_var.set("LLM model ready")
        messagebox.showinfo("LLM model", f"Model is ready at:\n{model_path}")

    # ------------------------------------------------------------------
    # Voice threshold calibration
    # ------------------------------------------------------------------

    def _auto_adjust_voice_threshold_clicked(
        self, voice_threshold_var: tk.StringVar, audio_device_value: str
    ) -> None:
        self.status_var.set("Calibrating voice threshold... speak normally for 3 seconds")
        threading.Thread(
            target=self._auto_adjust_voice_threshold_worker,
            args=(voice_threshold_var, audio_device_value),
            daemon=True,
        ).start()

    def _auto_adjust_voice_threshold_worker(
        self, voice_threshold_var: tk.StringVar, audio_device_value: str
    ) -> None:
        try:
            threshold = self._measure_voice_threshold(audio_device_value=audio_device_value)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Voice threshold calibration failed")
            self.root.after(0, self._on_auto_adjust_voice_threshold_done, voice_threshold_var, None, str(exc))
            return
        self.root.after(0, self._on_auto_adjust_voice_threshold_done, voice_threshold_var, threshold, "")

    def _on_auto_adjust_voice_threshold_done(
        self,
        voice_threshold_var: tk.StringVar,
        threshold: float | None,
        error: str,
    ) -> None:
        if error:
            self.status_var.set("Voice threshold calibration failed")
            messagebox.showerror("Voice threshold", f"Auto adjust failed:\n{error}")
            return
        assert threshold is not None
        formatted = f"{threshold:.4f}"
        voice_threshold_var.set(formatted)
        self.voice_threshold_var.set(formatted)
        self.asr_defaults["voice_threshold"] = threshold
        self.continuous_listener.voice_threshold = threshold
        self.status_var.set(f"Voice threshold calibrated: {formatted}")

    def _measure_voice_threshold(self, audio_device_value: str, duration_s: float = 3.0) -> float:
        frames = max(1, int(self.recorder.config.sample_rate_hz * duration_s))
        kwargs: dict = {
            "frames": frames,
            "samplerate": self.recorder.config.sample_rate_hz,
            "channels": self.recorder.config.channels,
            "dtype": "float32",
            "blocking": True,
        }
        device = parse_device_choice(audio_device_value)
        if device is not None:
            kwargs["device"] = device

        sample = sd.rec(**kwargs)
        audio = np.asarray(sample, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.01

        chunk = max(1, int(self.recorder.config.sample_rate_hz * 0.03))
        energies = [
            float(np.mean(np.abs(audio[start : start + chunk])))
            for start in range(0, int(audio.size), chunk)
            if audio[start : start + chunk].size > 0
        ]
        if not energies:
            return 0.01

        energies_np = np.asarray(energies, dtype=np.float32)
        noise_floor = float(np.percentile(energies_np, 20))
        speech_level = float(np.percentile(energies_np, 90))
        if speech_level <= 0:
            return 0.01

        threshold = noise_floor + (speech_level - noise_floor) * 0.35
        threshold = max(0.003, min(threshold, speech_level * 0.8, 0.1))
        return round(float(threshold), 4)

    # ------------------------------------------------------------------
    # Static format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_download_error(error: str) -> str:
        raw = (error or "").strip()
        if not raw:
            return "Unknown error"
        if "huggingface_hub_not_installed" in raw:
            return (
                "Downloader component (huggingface_hub) is missing in this build.\n"
                "Please install a newer installer build that includes downloader dependencies."
            )
        if "qwen_asr_not_installed" in raw:
            return (
                "Qwen ASR dependencies are missing in this build.\n"
                "Please install dependencies with 'pip install -r requirements.txt'."
            )
        if "openvino_export_dependencies_not_installed" in raw:
            return (
                "Whisper IR conversion dependencies are missing in this build.\n"
                "Please install dependencies with 'pip install -r requirements.txt'."
            )
        if "unsupported_asr_model" in raw:
            return (
                "Unsupported ASR model was selected.\n"
                "Choose one of the supported ASR models in Properties."
            )
        if "model_not_found_and_auto_download_disabled" in raw:
            return (
                "Model was not found locally and auto-download is disabled.\n"
                "Use 'Download LLM Model' or enable auto-download in settings."
            )
        if "model_download_failed" in raw:
            return (
                "ASR model download failed.\n"
                "Please check network/proxy/firewall settings and try again."
            )
        return raw

    @staticmethod
    def _directory_size_bytes(path: Path | None) -> int:
        if path is None or not path.exists():
            return 0
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        total = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        total += file_path.stat().st_size
                    except OSError:
                        continue
        except OSError:
            return 0
        return total

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        size = float(max(0, size_bytes))
        units = ("B", "KB", "MB", "GB", "TB")
        unit_idx = 0
        while size >= 1024.0 and unit_idx < len(units) - 1:
            size /= 1024.0
            unit_idx += 1
        if unit_idx == 0:
            return f"{int(size)} {units[unit_idx]}"
        return f"{size:.1f} {units[unit_idx]}"

    @staticmethod
    def _format_elapsed(elapsed_s: int) -> str:
        minutes, seconds = divmod(max(0, elapsed_s), 60)
        return f"{minutes:02d}:{seconds:02d}"
