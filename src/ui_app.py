import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from .asr import ASREngine
from .audio_capture import AudioConfig, AudioRecorder
from .storage import Storage
from .text_processing import ProcessOptions, process_text


class VoiceInputApp:
    def __init__(
        self,
        root: tk.Tk,
        asr_engine: ASREngine,
        recorder: AudioRecorder,
        storage: Storage,
        rules: dict,
    ):
        self.root = root
        self.asr_engine = asr_engine
        self.recorder = recorder
        self.storage = storage
        self.rules = rules
        self.logger = logging.getLogger(__name__)

        self.auto_edit_var = tk.BooleanVar(value=True)
        self.remove_fillers_var = tk.BooleanVar(value=True)
        self.remove_habits_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Starting...")
        self.mic_var = tk.StringVar(value="Mic: Idle")
        self.current_raw_text = ""

        self._build_ui()
        self._load_initial_state()

    def _build_ui(self) -> None:
        self.root.title("PROCESSING")
        self.root.geometry("860x820")
        self.root.configure(bg="#0a0e14")

        container = tk.Frame(self.root, bg="#0a0e14")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        top_bar = tk.Frame(container, bg="#141b26", highlightthickness=1, highlightbackground="#273142")
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text="●", fg="#ff9f1c", bg="#141b26", font=("Consolas", 11, "bold")).pack(
            side=tk.LEFT, padx=(10, 6), pady=8
        )
        tk.Label(
            top_bar,
            text="PROCESSING",
            fg="#e6edf3",
            bg="#141b26",
            font=("Consolas", 11, "bold"),
        ).pack(side=tk.LEFT, pady=8)
        tk.Label(
            top_bar,
            textvariable=self.status_var,
            fg="#9fb1c7",
            bg="#141b26",
            font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=12, pady=8)

        shortcut = tk.Frame(container, bg="#0a0e14")
        shortcut.pack(fill=tk.X, pady=(10, 0))
        tk.Label(
            shortcut,
            text="Shortcut: HotKey (mods: Modifiers(CONTROL), key: Space)",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9),
        ).pack(fill=tk.X)
        tk.Label(
            shortcut,
            textvariable=self.mic_var,
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9),
        ).pack(fill=tk.X)

        controls = tk.Frame(container, bg="#0a0e14")
        controls.pack(fill=tk.X, pady=(8, 4))

        self.record_button = tk.Button(
            controls,
            text="Toggle Recording",
            command=self.toggle_recording,
            bg="#1f6feb",
            fg="#ffffff",
            activebackground="#2f81f7",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Consolas", 10, "bold"),
            cursor="hand2",
        )
        self.record_button.pack(side=tk.LEFT)

        tk.Checkbutton(
            controls,
            text="Auto Edit",
            variable=self.auto_edit_var,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=(12, 4))
        tk.Checkbutton(
            controls,
            text="Remove Fillers",
            variable=self.remove_fillers_var,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            controls,
            text="Remove Habits",
            variable=self.remove_habits_var,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=4)

        log_frame = tk.Frame(container, bg="#0a0e14")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 8))
        self.log_text = tk.Text(
            log_frame,
            height=17,
            wrap=tk.WORD,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            padx=8,
            pady=8,
            font=("Consolas", 10),
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scroll.set, state=tk.DISABLED)

        final_title = tk.Label(
            container,
            text="FINAL RESULT",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9, "bold"),
        )
        final_title.pack(fill=tk.X)

        self.final_text = tk.Text(
            container,
            height=14,
            wrap=tk.WORD,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.final_text.pack(fill=tk.BOTH, expand=True)
        self._log("Starting Recording")

    def _load_initial_state(self) -> None:
        auto = self.storage.load_autosave()
        if auto:
            self.current_raw_text = auto.raw_text
            self.final_text.insert("1.0", auto.final_text)
            self._log("Loaded autosave from last session")
        self.status_var.set("Ready")

    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            try:
                self.recorder.start()
                self.record_button.config(text="Stop Recording", bg="#b62324", activebackground="#d73a49")
                self.status_var.set("Recording")
                self.mic_var.set("Mic: Live")
                self._log("Toggle: Started Recording")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("録音エラー", str(exc))
                self.logger.exception("Failed to start recording")
                self._log(f"Recording start failed: {exc}")
            return

        self.record_button.config(text="Toggle Recording", bg="#1f6feb", activebackground="#2f81f7")
        self.status_var.set("Transcribing")
        self.mic_var.set("Mic: Idle")
        self._log("Toggle: Stopping Recording")

        audio = self.recorder.stop()
        seconds = audio.size / max(1, self.recorder.config.sample_rate_hz)
        self._log(f"Recording saved to memory ({seconds:.2f}s)")
        self._log(f"Audio size: {audio.nbytes} bytes")
        threading.Thread(target=self._transcribe_and_process, args=(audio,), daemon=True).start()

    def _transcribe_and_process(self, audio_data) -> None:  # noqa: ANN001
        try:
            self.root.after(0, self._log, "Transcribing (v2)...")
            raw = self.asr_engine.transcribe(audio_data)
            self.root.after(0, self._log, "Reading audio file complete")
            options = ProcessOptions(
                auto_edit=self.auto_edit_var.get(),
                remove_fillers=self.remove_fillers_var.get(),
                remove_habits=self.remove_habits_var.get(),
            )
            self.root.after(0, self._log, "Post-processing...")
            final = process_text(raw, self.rules, options)
            self.storage.save_autosave(raw, final)
            self.storage.append_history(raw, final)
            self.root.after(0, self._log, "Sending text to local formatter...")
            self.root.after(0, self._apply_results, raw, final, "")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            self.root.after(0, self._apply_results, "", "", str(exc))

    def _apply_results(self, raw: str, final: str, error: str) -> None:
        if error:
            self.status_var.set("Error")
            self._log(f"ERROR: {error}")
            messagebox.showerror("処理エラー", error)
            return

        self._set_text(self.final_text, final)
        self.current_raw_text = raw
        self.status_var.set("Done")
        self._log(f"Transcribed: {raw[:80]}..." if raw else "Transcribed: (empty)")
        self._log("Processing complete")

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    def _log(self, message: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


def build_app(
    root: tk.Tk,
    model_dir: Path,
    audio_config: AudioConfig,
    storage: Storage,
    rules: dict,
) -> VoiceInputApp:
    engine = ASREngine(model_dir=model_dir, sample_rate_hz=audio_config.sample_rate_hz)
    recorder = AudioRecorder(config=audio_config)
    return VoiceInputApp(root=root, asr_engine=engine, recorder=recorder, storage=storage, rules=rules)
