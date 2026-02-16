import logging
import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import messagebox

from .asr import ASREngine
from .audio_capture import AudioConfig, AudioRecorder
from .business_email import to_business_email
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
        self.business_email_var = tk.BooleanVar(value=False)
        self.system_wide_input_var = tk.BooleanVar(value=enable_system_wide_input_default)
        self.status_var = tk.StringVar(value="Starting...")
        self.current_raw_text = ""
        self.hotkey_pressed = False
        self.llm_enabled_var = tk.BooleanVar(value=bool(self.llm_defaults.get("enabled", True)))
        self.properties_window: tk.Toplevel | None = None

        self.system_wide_input = SystemWideInput(
            dispatch_on_ui=lambda cb: self.root.after(0, cb),
            on_toggle=self.toggle_recording,
        )

        self._build_ui()
        self._bind_hotkeys()
        self._bind_context_menu()
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

        tk.Label(
            controls,
            text="Right-click to open Properties",
            bg="#0a0e14",
            fg="#8b9fb6",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=(12, 4))

        system_frame = tk.Frame(container, bg="#0a0e14")
        system_frame.pack(fill=tk.X, pady=(0, 8))
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

    def _bind_context_menu(self) -> None:
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Properties...", command=self._open_properties_dialog)
        self.root.bind("<Button-3>", self._show_context_menu)
        self.root.bind("<Control-Button-1>", self._show_context_menu)

    def _show_context_menu(self, event):  # noqa: ANN001
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _open_properties_dialog(self) -> None:
        if self.properties_window is not None and self.properties_window.winfo_exists():
            self.properties_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Properties")
        win.geometry("360x280")
        win.resizable(False, False)
        win.transient(self.root)
        self.properties_window = win

        auto_edit_var = tk.BooleanVar(value=self.auto_edit_var.get())
        remove_fillers_var = tk.BooleanVar(value=self.remove_fillers_var.get())
        remove_habits_var = tk.BooleanVar(value=self.remove_habits_var.get())
        business_email_var = tk.BooleanVar(value=self.business_email_var.get())
        system_wide_var = tk.BooleanVar(value=self.system_wide_input_var.get())
        llm_enabled_var = tk.BooleanVar(value=self.llm_enabled_var.get())

        frame = tk.Frame(win, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Checkbutton(frame, text="Auto edit", variable=auto_edit_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Remove fillers", variable=remove_fillers_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Remove habits", variable=remove_habits_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Convert to business email", variable=business_email_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Enable LLM correction", variable=llm_enabled_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(
            frame,
            text="System-wide input (paste to active app on completion)",
            variable=system_wide_var,
        ).pack(anchor=tk.W, pady=4)
        tk.Button(
            frame,
            text="Download LLM Model",
            command=self._download_model_clicked,
            bg="#2ea043",
            fg="#ffffff",
            activebackground="#3fb950",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            font=("Consolas", 9, "bold"),
            cursor="hand2",
        ).pack(anchor=tk.W, pady=(10, 0))

        def apply_and_close() -> None:
            self.auto_edit_var.set(auto_edit_var.get())
            self.remove_fillers_var.set(remove_fillers_var.get())
            self.remove_habits_var.set(remove_habits_var.get())
            self.business_email_var.set(business_email_var.get())
            self.llm_enabled_var.set(llm_enabled_var.get())
            self.llm_defaults["enabled"] = bool(llm_enabled_var.get())

            before = self.system_wide_input_var.get()
            after = system_wide_var.get()
            self.system_wide_input_var.set(after)
            if before != after:
                self._toggle_system_wide_input()
            self.status_var.set("Properties updated")
            self.properties_window = None
            win.destroy()

        buttons = tk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(16, 0))
        tk.Button(buttons, text="Apply", command=apply_and_close, width=10).pack(side=tk.LEFT)
        tk.Button(buttons, text="Cancel", command=win.destroy, width=10).pack(side=tk.RIGHT)

        def on_close() -> None:
            self.properties_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

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

    def _download_model_clicked(self) -> None:
        self.status_var.set("Downloading LLM model...")
        threading.Thread(target=self._download_model_worker, daemon=True).start()

    def _download_model_worker(self) -> None:
        try:
            model_path = self.llm_editor.download_model()
            self.root.after(0, self._on_download_model_done, model_path, "")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Model download failed")
            self.root.after(0, self._on_download_model_done, "", str(exc))

    def _on_download_model_done(self, model_path: str, error: str) -> None:
        if error:
            self.status_var.set("Model download failed")
            messagebox.showerror("LLM model download error", error)
            return
        self.status_var.set("LLM model ready")
        messagebox.showinfo("LLM model", f"Model is ready at:\n{model_path}")

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
        pipeline_started = time.perf_counter()
        timings: dict[str, int] = {}
        try:
            started = time.perf_counter()
            raw_asr = self.asr_engine.transcribe(audio_data)
            timings["asr"] = int((time.perf_counter() - started) * 1000)

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
                raw,
                final,
                llm_applied=llm_result.applied,
                llm_latency_ms=llm_result.latency_ms,
                fallback_reason=llm_result.fallback_reason,
                processing_total_ms=total_ms,
                processing_breakdown_ms=timings,
            )
            self.storage.append_history(
                raw,
                final,
                llm_applied=llm_result.applied,
                llm_latency_ms=llm_result.latency_ms,
                fallback_reason=llm_result.fallback_reason,
                processing_total_ms=total_ms,
                processing_breakdown_ms=timings,
            )
            timings["storage"] = int((time.perf_counter() - started) * 1000)

            self.logger.info("Pipeline timings (ms): %s", timings)
            self.root.after(0, self._apply_results, raw, final, "", llm_result.fallback_reason, timings)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            self.root.after(0, self._apply_results, "", "", str(exc), "", timings)

    def _apply_results(
        self,
        raw: str,
        final: str,
        error: str,
        fallback_reason: str = "",
        timings: dict[str, int] | None = None,
    ) -> None:
        if error:
            self.status_var.set("Error")
            messagebox.showerror("Processing error", error)
            return

        timing_suffix = self._format_timing_suffix(timings)
        self._set_text(self.final_text, final)
        self.current_raw_text = raw
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

    def _on_close(self) -> None:
        self.system_wide_input.stop()
        self.root.destroy()

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    @staticmethod
    def _format_timing_suffix(timings: dict[str, int] | None) -> str:
        if not timings:
            return ""

        ordered_keys = ["total", "asr", "rules", "llm", "storage"]
        labels = {
            "total": "total",
            "asr": "asr",
            "rules": "rules",
            "llm": "llm",
            "storage": "save",
        }
        parts = [f"{labels[key]} {timings[key]}ms" for key in ordered_keys if key in timings]
        return f" [{', '.join(parts)}]" if parts else ""


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
        llm_device=str(llm_defaults.get("device", "GPU")),
        auto_download=bool(llm_defaults.get("auto_download", False)),
        download_dir=Path(str(llm_defaults.get("download_dir", "models/openvino"))),
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
