"""RecordingMixin: recording control, transcription pipeline, continuous mode.

Mixed into VoiceInputApp. Accesses self.* attributes set in VoiceInputApp.__init__.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox

from .business_email import to_business_email
from .llm_post_editor import LLMOptions
from .text_processing import ProcessOptions, process_text
from .voice_commands import detect_voice_command


class RecordingMixin:
    # ------------------------------------------------------------------
    # Recording toggle
    # ------------------------------------------------------------------

    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            try:
                self.recorder.start()
                self.record_button.config(state=tk.NORMAL)
                self.record_button.config(text="Stop Recording", bg="#b62324", activebackground="#d73a49")
                self.status_var.set("Recording")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Recording error", str(exc))
                self.logger.exception("Failed to start recording")
            return

        self.record_button.config(
            text="Start Recording", bg="#1f6feb", activebackground="#2f81f7", state=tk.DISABLED
        )
        self._start_processing_indicator("Stopping")
        threading.Thread(target=self._stop_and_process_worker, daemon=True).start()

    def _stop_and_process_worker(self) -> None:
        try:
            audio = self.recorder.stop()
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to stop recording")
            self.root.after(0, self._apply_results, "", "", str(exc), "", {})
            return

        self.root.after(0, self._set_processing_phase, "Transcribing")
        self._transcribe_and_process(audio)

    # ------------------------------------------------------------------
    # Continuous mode
    # ------------------------------------------------------------------

    def _toggle_continuous(self) -> None:
        if self._continuous_mode_active:
            self._stop_continuous()
        else:
            self._start_continuous()

    def _start_continuous(self) -> None:
        try:
            self.continuous_listener.start()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Continuous mode error", str(exc))
            self.logger.exception("Failed to start continuous listener")
            return
        self._continuous_mode_active = True
        self.record_button.config(state=tk.DISABLED)
        if self.continuous_button is not None:
            self.continuous_button.config(text="Stop Continuous", bg="#b62324", activebackground="#d73a49")
        self.status_var.set("Listening...")

    def _stop_continuous(self) -> None:
        self.continuous_listener.stop()
        self._continuous_mode_active = False
        self.record_button.config(state=tk.NORMAL)
        if self.continuous_button is not None:
            self.continuous_button.config(text="Continuous", bg="#2ea043", activebackground="#3fb950")
        self.status_var.set("Ready (Ctrl+Space / Ctrl+Shift+Space / Ctrl+Alt+Space)")

    def _on_continuous_utterance(self, audio_data) -> None:  # noqa: ANN001
        """Called from ContinuousListener background thread when an utterance ends."""
        if not self._processing_lock.acquire(blocking=False):
            self.root.after(0, self.status_var.set, "Listening... (busy, skipped)")
            return
        try:
            self.root.after(0, self._start_processing_indicator, "Transcribing")
            self._transcribe_and_process(audio_data)
        finally:
            self._processing_lock.release()
            if self._continuous_mode_active:
                self.root.after(0, self.status_var.set, "Listening...")

    # ------------------------------------------------------------------
    # Transcription pipeline
    # ------------------------------------------------------------------

    def _transcribe_and_process(self, audio_data) -> None:  # noqa: ANN001
        pipeline_started = time.perf_counter()
        timings: dict[str, int] = {}
        try:
            started = time.perf_counter()
            raw_asr = self.asr_engine.transcribe(audio_data)
            timings["asr"] = int((time.perf_counter() - started) * 1000)

            if not raw_asr and self._continuous_mode_active:
                self.root.after(0, self._stop_processing_indicator)
                return

            cmd = detect_voice_command(raw_asr)
            if cmd is not None:
                self.root.after(0, self._execute_voice_command, cmd)
                return

            started = time.perf_counter()
            raw = self.personal_dictionary.apply(raw_asr)
            timings["dictionary"] = int((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            process_result = process_text(
                raw,
                self.rules,
                ProcessOptions(
                    auto_edit=self.auto_edit_var.get(),
                    remove_fillers=self.remove_fillers_var.get(),
                    remove_habits=self.remove_habits_var.get(),
                ),
            )
            timings["rules"] = int((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            llm_result = self.llm_editor.refine(
                raw_text=raw_asr,
                preprocessed_text=process_result.final_text,
                options=LLMOptions(
                    enabled=bool(self.llm_enabled_var.get()),
                    strength=str(self.llm_defaults.get("strength", "medium")),
                    max_input_chars=int(self.llm_defaults.get("max_input_chars", 1200)),
                    max_change_ratio=float(self.llm_defaults.get("max_change_ratio", 0.35)),
                    domain_hint=str(self.llm_defaults.get("domain_hint", "")),
                ),
            )
            timings["llm"] = int((time.perf_counter() - started) * 1000)

            final = llm_result.final_text
            if self.business_email_var.get():
                started = time.perf_counter()
                final = to_business_email(final)
                timings["business_email"] = int((time.perf_counter() - started) * 1000)

            total_ms = int((time.perf_counter() - pipeline_started) * 1000)
            timings["total"] = total_ms

            started = time.perf_counter()
            self.storage.save_autosave(
                raw, final,
                llm_applied=llm_result.applied,
                llm_latency_ms=llm_result.latency_ms,
                fallback_reason=llm_result.fallback_reason,
                processing_total_ms=total_ms,
                processing_breakdown_ms=timings,
            )
            self.storage.append_history(
                raw, final,
                llm_applied=llm_result.applied,
                llm_latency_ms=llm_result.latency_ms,
                fallback_reason=llm_result.fallback_reason,
                processing_total_ms=total_ms,
                processing_breakdown_ms=timings,
            )
            timings["storage"] = int((time.perf_counter() - started) * 1000)

            self.logger.info("Pipeline timings (ms): %s", timings)
            self.root.after(0, self._apply_results, raw_asr, final, "", llm_result.fallback_reason, timings)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            self.root.after(0, self._apply_results, "", "", str(exc), "", timings)

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _apply_results(
        self,
        asr_text_value: str,
        final: str,
        error: str,
        fallback_reason: str = "",
        timings: dict[str, int] | None = None,
    ) -> None:
        self._stop_processing_indicator()
        if not self._continuous_mode_active:
            self.record_button.config(state=tk.NORMAL)

        if error:
            self.status_var.set("Error" if not self._continuous_mode_active else f"Error: {error[:60]}")
            if not self._continuous_mode_active:
                messagebox.showerror("Processing error", self._format_processing_error(error))
            return

        timing_suffix = self._format_timing_suffix(timings)
        self._set_text(self.final_text, final)
        if self.asr_text is not None:
            self._set_text(self.asr_text, asr_text_value)
        self.current_raw_text = asr_text_value
        if self.system_wide_input_var.get():
            try:
                self.system_wide_input.paste_to_active_app(final)
                if fallback_reason and fallback_reason not in {"", "disabled"}:
                    self.status_var.set(f"Done (fallback: {fallback_reason}){timing_suffix}")
                else:
                    self.status_var.set(f"Done (pasted to active app){timing_suffix}")
            except Exception as exc:  # noqa: BLE001
                self.status_var.set(f"Done (paste failed){timing_suffix}")
                messagebox.showwarning("Paste failed", str(exc))
        else:
            if fallback_reason and fallback_reason not in {"", "disabled"}:
                self.status_var.set(f"Done (fallback: {fallback_reason}){timing_suffix}")
            else:
                self.status_var.set(f"Done{timing_suffix}")

    # ------------------------------------------------------------------
    # Processing indicator
    # ------------------------------------------------------------------

    def _start_processing_indicator(self, phase: str) -> None:
        self._processing_active = True
        self._processing_started = time.perf_counter()
        self._processing_phase = phase
        self._processing_tick_token += 1
        token = self._processing_tick_token
        self._tick_processing_indicator(token)

    def _set_processing_phase(self, phase: str) -> None:
        self._processing_phase = phase

    def _stop_processing_indicator(self) -> None:
        self._processing_active = False
        self._processing_tick_token += 1

    def _tick_processing_indicator(self, token: int) -> None:
        if not self._processing_active or token != self._processing_tick_token:
            return
        elapsed = int(time.perf_counter() - self._processing_started)
        dots = "." * ((elapsed % 3) + 1)
        self.status_var.set(f"{self._processing_phase}{dots} ({elapsed}s)")
        self.root.after(250, self._tick_processing_indicator, token)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    @staticmethod
    def _format_timing_suffix(timings: dict[str, int] | None) -> str:
        if not timings:
            return ""
        ordered_keys = ["total", "asr", "rules", "llm", "storage"]
        labels = {"total": "total", "asr": "asr", "rules": "rules", "llm": "llm", "storage": "save"}
        parts = [f"{labels[key]} {timings[key]}ms" for key in ordered_keys if key in timings]
        return f" [{', '.join(parts)}]" if parts else ""

    @staticmethod
    def _format_processing_error(error: str) -> str:
        raw = (error or "").strip()
        normalized = raw.lower()
        if "asr_empty_output" in normalized:
            return (
                "ASR could not produce text from this audio.\n"
                "Check microphone input level and ASR model readiness, then retry."
            )
        if "asr_failed_all_windows" in normalized:
            return (
                "ASR failed on all audio windows.\n"
                "Try a shorter recording and switch ASR device (gpu/cpu) in Properties."
            )
        if "qwen_asr_not_installed" in normalized:
            return "Qwen ASR backend is not installed. Run: pip install -r requirements.txt"
        if "openvino_export_dependencies_not_installed" in normalized:
            return "Whisper IR conversion dependencies are missing. Run: pip install -r requirements.txt"
        if "torch_not_installed" in normalized:
            return "PyTorch is not installed. Run: pip install -r requirements.txt"
        if "vector too long" in raw.lower():
            return (
                "Audio segment is too long for one-pass transcription.\n"
                "Please try a shorter recording segment and retry."
            )
        return raw or "Unknown error"
