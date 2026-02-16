import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from .asr import ASREngine
from .audio_capture import AudioConfig, AudioRecorder
from .llm_post_editor import LLMOptions, LLMPostEditor
from .personal_dictionary import PersonalDictionary
from .storage import Storage
from .system_wide_input import SystemWideInput
from .text_processing import ProcessOptions, process_text


class VoiceInputApp:
    def __init__(
        self,
        root: tk.Tk,
        asr_engine: ASREngine,
        recorder: AudioRecorder,
        storage: Storage,
        rules: dict,
        personal_dictionary: PersonalDictionary,
        llm_editor: LLMPostEditor,
        llm_defaults: dict,
        enable_system_wide_input_default: bool,
    ):
        self.root = root
        self.asr_engine = asr_engine
        self.recorder = recorder
        self.storage = storage
        self.rules = rules
        self.personal_dictionary = personal_dictionary
        self.llm_editor = llm_editor
        self.llm_defaults = llm_defaults
        self.logger = logging.getLogger(__name__)

        self.auto_edit_var = tk.BooleanVar(value=True)
        self.remove_fillers_var = tk.BooleanVar(value=True)
        self.remove_habits_var = tk.BooleanVar(value=True)
        self.system_wide_input_var = tk.BooleanVar(value=enable_system_wide_input_default)
        self.status_var = tk.StringVar(value="Starting...")
        self.current_raw_text = ""
        self.hotkey_pressed = False

        self.system_wide_input = SystemWideInput(
            dispatch_on_ui=lambda cb: self.root.after(0, cb),
            on_toggle=self.toggle_recording,
        )

        self._build_ui()
        self._bind_hotkeys()
        self._load_initial_state()
        self._refresh_dictionary_list()

        if self.system_wide_input_var.get():
            self.system_wide_input.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.title("Voice Input App")
        self.root.geometry("430x840")
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
            text="Voice Input",
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

        controls = tk.Frame(container, bg="#0a0e14")
        controls.pack(fill=tk.X, pady=(10, 8))

        self.record_button = tk.Button(
            controls,
            text="Start Recording",
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
            text="Auto edit",
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
            text="Remove fillers",
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
            text="Remove habits",
            variable=self.remove_habits_var,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=4)

        system_frame = tk.Frame(container, bg="#0a0e14")
        system_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Checkbutton(
            system_frame,
            text="System-wide input (paste to active app on completion)",
            variable=self.system_wide_input_var,
            command=self._toggle_system_wide_input,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(anchor=tk.W)
        tk.Label(
            system_frame,
            text="Global hotkey: Ctrl+Shift+Space",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9),
        ).pack(fill=tk.X)

        dict_frame = tk.Frame(container, bg="#0a0e14", highlightthickness=1, highlightbackground="#273142")
        dict_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            dict_frame,
            text="Personal Dictionary (reading -> surface)",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9, "bold"),
        ).pack(fill=tk.X, padx=6, pady=(6, 2))

        form = tk.Frame(dict_frame, bg="#0a0e14")
        form.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(form, text="Reading", fg="#c9d1d9", bg="#0a0e14", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_reading_entry = tk.Entry(
            form,
            width=10,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
        )
        self.dict_reading_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(form, text="Surface", fg="#c9d1d9", bg="#0a0e14", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_surface_entry = tk.Entry(
            form,
            width=10,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
        )
        self.dict_surface_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Button(
            form,
            text="Add",
            command=self._add_dictionary_entry,
            bg="#2ea043",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
        ).pack(side=tk.LEFT)
        tk.Button(
            form,
            text="Remove",
            command=self._remove_dictionary_entry,
            bg="#b62324",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.dict_list = tk.Listbox(
            dict_frame,
            height=4,
            bg="#0b111a",
            fg="#dbe6f3",
            selectbackground="#1f6feb",
            selectforeground="#ffffff",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.dict_list.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.dict_list.bind("<<ListboxSelect>>", self._on_dictionary_selected)

        final_title = tk.Label(
            container,
            text="Final text",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9, "bold"),
        )
        final_title.pack(fill=tk.X)

        self.final_text = tk.Text(
            container,
            height=18,
            wrap=tk.WORD,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.final_text.pack(fill=tk.BOTH, expand=True)

    def _load_initial_state(self) -> None:
        auto = self.storage.load_autosave()
        if auto:
            self.current_raw_text = auto.raw_text
            self.final_text.insert("1.0", auto.final_text)
        self.status_var.set("Ready (Ctrl+Space / Ctrl+Shift+Space)")

    def _bind_hotkeys(self) -> None:
        self.root.bind_all("<Control-KeyPress-space>", self._on_hotkey_press)
        self.root.bind_all("<Control-KeyRelease-space>", self._on_hotkey_release)

    def _on_hotkey_press(self, event):  # noqa: ANN001
        if event.state & 0x0001:
            return
        if self.hotkey_pressed:
            return "break"
        self.hotkey_pressed = True
        self.toggle_recording()
        return "break"

    def _on_hotkey_release(self, event):  # noqa: ANN001, ARG002
        self.hotkey_pressed = False
        return "break"

    def _toggle_system_wide_input(self) -> None:
        if self.system_wide_input_var.get():
            self.system_wide_input.start()
            self.status_var.set("System-wide input: ON")
        else:
            self.system_wide_input.stop()
            self.status_var.set("System-wide input: OFF")

    def _refresh_dictionary_list(self) -> None:
        self.dict_entries = self.personal_dictionary.list_entries()
        self.dict_list.delete(0, tk.END)
        for item in self.dict_entries:
            self.dict_list.insert(tk.END, f"{item.reading} -> {item.surface} ({item.count})")

    def _on_dictionary_selected(self, event):  # noqa: ANN001
        if not self.dict_list.curselection():
            return
        idx = self.dict_list.curselection()[0]
        item = self.dict_entries[idx]
        self.dict_reading_entry.delete(0, tk.END)
        self.dict_reading_entry.insert(0, item.reading)
        self.dict_surface_entry.delete(0, tk.END)
        self.dict_surface_entry.insert(0, item.surface)

    def _add_dictionary_entry(self) -> None:
        try:
            self.personal_dictionary.add_or_update(
                reading=self.dict_reading_entry.get(),
                surface=self.dict_surface_entry.get(),
            )
        except ValueError as exc:
            messagebox.showwarning("Input missing", str(exc))
            return
        self._refresh_dictionary_list()
        self.status_var.set("Dictionary updated")

    def _remove_dictionary_entry(self) -> None:
        reading = self.dict_reading_entry.get().strip()
        if not reading:
            messagebox.showwarning("No target", "Please select a reading to remove.")
            return
        self.personal_dictionary.remove(reading)
        self._refresh_dictionary_list()
        self.status_var.set("Dictionary removed")

    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            try:
                self.recorder.start()
                self.record_button.config(text="Stop Recording", bg="#b62324", activebackground="#d73a49")
                self.status_var.set("Recording")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Recording error", str(exc))
                self.logger.exception("Failed to start recording")
            return

        self.record_button.config(text="Start Recording", bg="#1f6feb", activebackground="#2f81f7")
        self.status_var.set("Transcribing")

        audio = self.recorder.stop()
        threading.Thread(target=self._transcribe_and_process, args=(audio,), daemon=True).start()

    def _transcribe_and_process(self, audio_data) -> None:  # noqa: ANN001
        try:
            raw_asr = self.asr_engine.transcribe(audio_data)
            raw = self.personal_dictionary.apply(raw_asr)
            process_result = process_text(
                raw,
                self.rules,
                ProcessOptions(
                    auto_edit=self.auto_edit_var.get(),
                    remove_fillers=self.remove_fillers_var.get(),
                    remove_habits=self.remove_habits_var.get(),
                ),
            )

            llm_result = self.llm_editor.refine(
                raw_text=raw_asr,
                preprocessed_text=process_result.final_text,
                options=LLMOptions(
                    enabled=bool(self.llm_defaults.get("enabled", True)),
                    strength=str(self.llm_defaults.get("strength", "medium")),
                    max_input_chars=int(self.llm_defaults.get("max_input_chars", 1200)),
                    max_change_ratio=float(self.llm_defaults.get("max_change_ratio", 0.35)),
                    domain_hint=str(self.llm_defaults.get("domain_hint", "")),
                ),
            )

            final = llm_result.final_text
            self.storage.save_autosave(
                raw,
                final,
                llm_applied=llm_result.applied,
                llm_latency_ms=llm_result.latency_ms,
                fallback_reason=llm_result.fallback_reason,
            )
            self.storage.append_history(
                raw,
                final,
                llm_applied=llm_result.applied,
                llm_latency_ms=llm_result.latency_ms,
                fallback_reason=llm_result.fallback_reason,
            )
            self.root.after(0, self._apply_results, raw, final, "", llm_result.fallback_reason)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            self.root.after(0, self._apply_results, "", "", str(exc), "")

    def _apply_results(self, raw: str, final: str, error: str, fallback_reason: str = "") -> None:
        if error:
            self.status_var.set("Error")
            messagebox.showerror("Processing error", error)
            return

        self._set_text(self.final_text, final)
        self.current_raw_text = raw
        if self.system_wide_input_var.get():
            try:
                self.system_wide_input.paste_to_active_app(final)
                if fallback_reason and fallback_reason not in {"", "disabled"}:
                    self.status_var.set(f"Done (fallback: {fallback_reason})")
                else:
                    self.status_var.set("Done (pasted to active app)")
            except Exception as exc:  # noqa: BLE001
                self.status_var.set("Done (paste failed)")
                messagebox.showwarning("Paste failed", str(exc))
        else:
            if fallback_reason and fallback_reason not in {"", "disabled"}:
                self.status_var.set(f"Done (fallback: {fallback_reason})")
            else:
                self.status_var.set("Done")

    def _on_close(self) -> None:
        self.system_wide_input.stop()
        self.root.destroy()

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)


def build_app(
    root: tk.Tk,
    model_dir: Path,
    audio_config: AudioConfig,
    storage: Storage,
    rules: dict,
    personal_dictionary: PersonalDictionary,
    enable_system_wide_input_default: bool,
    llm_defaults: dict,
) -> VoiceInputApp:
    engine = ASREngine(model_dir=model_dir, sample_rate_hz=audio_config.sample_rate_hz)
    recorder = AudioRecorder(config=audio_config)
    llm_editor = LLMPostEditor(
        model_path=Path(str(llm_defaults.get("model_path", "OpenVINO/Qwen3-8B-int4-cw-ov"))),
        timeout_ms=int(llm_defaults.get("timeout_ms", 8000)),
        blocked_patterns=list(llm_defaults.get("blocked_patterns", [])),
        llm_device=str(llm_defaults.get("device", "CPU")),
    )
    return VoiceInputApp(
        root=root,
        asr_engine=engine,
        recorder=recorder,
        storage=storage,
        rules=rules,
        personal_dictionary=personal_dictionary,
        llm_editor=llm_editor,
        llm_defaults=llm_defaults,
        enable_system_wide_input_default=enable_system_wide_input_default,
    )
