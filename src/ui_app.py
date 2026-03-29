import logging
import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import messagebox
from tkinter import ttk

import sounddevice as sd

from .asr import ASREngine
from .audio_capture import AudioConfig, AudioRecorder
from .business_email import to_business_email
from .llm_post_editor import LLMOptions, LLMPostEditor
from .personal_dictionary import PersonalDictionary
from .storage import Storage
from .app_launcher import AppLauncher
from .continuous_listener import ContinuousListener
from .system_wide_input import SystemWideInput
from .text_processing import ProcessOptions, process_text
from .voice_commands import VoiceCommand, detect_voice_command

ASR_MODEL_CHOICES = (
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B",
)


def _get_input_device_choices() -> list[str]:
    """Return list of 'index: name' strings for all input devices."""
    choices = ["auto (system default)"]
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                choices.append(f"{i}: {dev['name']}")
    except Exception:  # noqa: BLE001
        pass
    return choices


def _parse_device_choice(value: str) -> "int | None":
    """Parse a device choice string back to an int index (or None for auto)."""
    v = (value or "").strip()
    if not v or v.startswith("auto"):
        return None
    try:
        return int(v.split(":")[0])
    except (ValueError, IndexError):
        return None


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
        asr_defaults: dict,
        root_dir: Path,
        enable_system_wide_input_default: bool,
    ):
        self.root = root
        self.root_dir = root_dir
        self.asr_engine = asr_engine
        self.recorder = recorder
        self.storage = storage
        self.rules = rules
        self.personal_dictionary = personal_dictionary
        self.llm_editor = llm_editor
        self.llm_defaults = llm_defaults
        self.asr_defaults = asr_defaults
        self.logger = logging.getLogger(__name__)

        self.auto_edit_var = tk.BooleanVar(value=True)
        self.remove_fillers_var = tk.BooleanVar(value=True)
        self.remove_habits_var = tk.BooleanVar(value=True)
        self.business_email_var = tk.BooleanVar(value=False)
        self.system_wide_input_var = tk.BooleanVar(value=enable_system_wide_input_default)
        self.status_var = tk.StringVar(value="Starting...")
        self.current_raw_text = ""
        self.hotkey_pressed = False
        self.llm_enabled_var = tk.BooleanVar(value=bool(self.llm_defaults.get("enabled", False)))
        self.whisper_model_name_var = tk.StringVar(
            value=str(self.asr_defaults.get("whisper_model_name", "Qwen/Qwen3-ASR-0.6B"))
        )
        self.whisper_device_var = tk.StringVar(value=str(self.asr_defaults.get("whisper_device", "auto")))
        self.whisper_compute_type_var = tk.StringVar(
            value=str(self.asr_defaults.get("whisper_compute_type", "int8"))
        )
        # Audio input device: store as "index: name" string or "auto (system default)"
        _saved_device = self.asr_defaults.get("audio_input_device", None)
        self.audio_device_var = tk.StringVar(value=str(_saved_device) if _saved_device else "auto (system default)")
        self.voice_threshold_var = tk.StringVar(
            value=str(self.asr_defaults.get("voice_threshold", "0.02"))
        )
        self.properties_window: tk.Toplevel | None = None
        self.asr_text: tk.Text | None = None
        self.dict_reading_entry: tk.Entry | None = None
        self.dict_surface_entry: tk.Entry | None = None
        self.dict_list: tk.Listbox | None = None
        self.dict_entries = []
        self._processing_active = False
        self._processing_lock = threading.Lock()
        self._processing_started = 0.0
        self._processing_phase = "Processing"
        self._processing_tick_token = 0
        self._continuous_mode_active = False
        self.continuous_button: tk.Button | None = None

        self.system_wide_input = SystemWideInput(
            dispatch_on_ui=lambda cb: self.root.after(0, cb),
            on_toggle=self.toggle_recording,
        )

        self.continuous_listener = ContinuousListener(
            config=self.recorder.config,
            on_utterance=self._on_continuous_utterance,
            on_voice_start=lambda: self.root.after(0, self.status_var.set, "Speaking..."),
            voice_threshold=float(self.voice_threshold_var.get()),
        )
        self.app_launcher = AppLauncher()

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

        self.continuous_button = tk.Button(
            controls,
            text="Continuous",
            command=self._toggle_continuous,
            bg="#2ea043",
            fg="#ffffff",
            activebackground="#3fb950",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Consolas", 10, "bold"),
            cursor="hand2",
        )
        self.continuous_button.pack(side=tk.LEFT, padx=(8, 0))

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
        tk.Label(
            system_frame,
            text="Voice cmds: クリア / コピー / プロパティ / 辞書登録 [読み] [表記] / 辞書削除 [読み]",
            fg="#5a7a9b",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 8),
        ).pack(fill=tk.X)

        output_title = tk.Label(
            container,
            text="Output",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9, "bold"),
        )
        output_title.pack(fill=tk.X)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Output.TNotebook", background="#0a0e14", borderwidth=0)
        style.configure(
            "Output.TNotebook.Tab",
            padding=(10, 4),
            font=("Consolas", 9, "bold"),
            foreground="#dbe6f3",
            background="#1a2433",
        )
        style.map(
            "Output.TNotebook.Tab",
            foreground=[("selected", "#ffffff")],
            background=[("selected", "#2f81f7")],
        )

        tabs = ttk.Notebook(container, style="Output.TNotebook")
        tabs.pack(fill=tk.BOTH, expand=True)

        asr_tab = tk.Frame(tabs, bg="#0a0e14")
        final_tab = tk.Frame(tabs, bg="#0a0e14")
        tabs.add(asr_tab, text="ASR Text")
        tabs.add(final_tab, text="Final")
        tab_selected_colors = {
            0: "#14532d",  # ASR Text
            1: "#1d4ed8",  # Final
        }

        def apply_selected_tab_color() -> None:
            try:
                current_idx = tabs.index("current")
            except tk.TclError:
                current_idx = 0
            selected_bg = tab_selected_colors.get(current_idx, "#2f81f7")
            style.map(
                "Output.TNotebook.Tab",
                foreground=[("selected", "#ffffff")],
                background=[("selected", selected_bg)],
            )

        tabs.bind("<<NotebookTabChanged>>", lambda _event: apply_selected_tab_color())
        apply_selected_tab_color()

        self.asr_text = tk.Text(
            asr_tab,
            height=18,
            wrap=tk.WORD,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.asr_text.pack(fill=tk.BOTH, expand=True)

        self.final_text = tk.Text(
            final_tab,
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
            self._set_text(self.final_text, auto.final_text)
            if self.asr_text is not None:
                self._set_text(self.asr_text, auto.raw_text)
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
        win.geometry("420x700")
        win.resizable(False, False)
        win.transient(self.root)
        self.properties_window = win

        auto_edit_var = tk.BooleanVar(value=self.auto_edit_var.get())
        remove_fillers_var = tk.BooleanVar(value=self.remove_fillers_var.get())
        remove_habits_var = tk.BooleanVar(value=self.remove_habits_var.get())
        business_email_var = tk.BooleanVar(value=self.business_email_var.get())
        system_wide_var = tk.BooleanVar(value=self.system_wide_input_var.get())
        llm_enabled_var = tk.BooleanVar(value=self.llm_enabled_var.get())
        whisper_model_name_var = tk.StringVar(value=self.whisper_model_name_var.get())
        whisper_device_var = tk.StringVar(value=self.whisper_device_var.get())
        whisper_compute_type_var = tk.StringVar(value=self.whisper_compute_type_var.get())
        audio_device_var = tk.StringVar(value=self.audio_device_var.get())
        voice_threshold_var = tk.StringVar(value=self.voice_threshold_var.get())

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
        tk.Label(frame, text="Audio input device").pack(anchor=tk.W, pady=(8, 0))
        device_choices = _get_input_device_choices()
        ttk.Combobox(
            frame,
            textvariable=audio_device_var,
            values=device_choices,
            state="readonly",
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(frame, text="Voice threshold (Continuous mode: increase if silence not detected, e.g. 0.01-0.1)").pack(anchor=tk.W, pady=(8, 0))
        tk.Entry(frame, textvariable=voice_threshold_var, width=10).pack(anchor=tk.W)
        tk.Label(frame, text="ASR model name").pack(anchor=tk.W, pady=(8, 0))
        ttk.Combobox(
            frame,
            textvariable=whisper_model_name_var,
            values=ASR_MODEL_CHOICES,
            state="normal",
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(frame, text="ASR device").pack(anchor=tk.W, pady=(8, 0))
        tk.OptionMenu(frame, whisper_device_var, "auto", "npu", "gpu", "cpu", "cuda").pack(anchor=tk.W, fill=tk.X)
        tk.Label(frame, text="ASR compute type").pack(anchor=tk.W, pady=(8, 0))
        tk.OptionMenu(
            frame,
            whisper_compute_type_var,
            "int8",
            "int8_float16",
            "float16",
            "float32",
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Button(
            frame,
            text="Download ASR Model",
            command=lambda: download_asr_model_from_dialog(),
            bg="#1f6feb",
            fg="#ffffff",
            activebackground="#2f81f7",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            font=("Consolas", 9, "bold"),
            cursor="hand2",
        ).pack(anchor=tk.W, pady=(8, 0))
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

        dict_frame = tk.Frame(frame, highlightthickness=1, highlightbackground="#273142")
        dict_frame.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            dict_frame,
            text="Personal Dictionary (reading -> surface)",
            anchor="w",
            font=("Consolas", 9, "bold"),
        ).pack(fill=tk.X, padx=6, pady=(6, 2))

        dict_form = tk.Frame(dict_frame)
        dict_form.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(dict_form, text="Reading", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_reading_entry = tk.Entry(dict_form, width=10, relief=tk.FLAT)
        self.dict_reading_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(dict_form, text="Surface", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_surface_entry = tk.Entry(dict_form, width=10, relief=tk.FLAT)
        self.dict_surface_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Button(
            dict_form,
            text="Add",
            command=self._add_dictionary_entry,
            bg="#2ea043",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
        ).pack(side=tk.LEFT)
        tk.Button(
            dict_form,
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
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.dict_list.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.dict_list.bind("<<ListboxSelect>>", self._on_dictionary_selected)
        self._refresh_dictionary_list()

        def download_asr_model_from_dialog() -> None:
            model_name = whisper_model_name_var.get().strip() or "Qwen/Qwen3-ASR-0.6B"
            device = whisper_device_var.get().strip() or "auto"
            compute_type = whisper_compute_type_var.get().strip() or "int8"
            self._download_asr_model_clicked(
                model_name=model_name,
                device=device,
                compute_type=compute_type,
            )

        def apply_and_close() -> None:
            self.auto_edit_var.set(auto_edit_var.get())
            self.remove_fillers_var.set(remove_fillers_var.get())
            self.remove_habits_var.set(remove_habits_var.get())
            self.business_email_var.set(business_email_var.get())
            self.llm_enabled_var.set(llm_enabled_var.get())
            self.llm_defaults["enabled"] = bool(llm_enabled_var.get())
            self.whisper_model_name_var.set(
                whisper_model_name_var.get().strip() or "Qwen/Qwen3-ASR-0.6B"
            )
            self.whisper_device_var.set(whisper_device_var.get())
            self.whisper_compute_type_var.set(whisper_compute_type_var.get())
            self._apply_asr_settings()

            # Apply audio input device
            chosen_device = audio_device_var.get()
            self.audio_device_var.set(chosen_device)
            self.asr_defaults["audio_input_device"] = chosen_device
            new_device_idx = _parse_device_choice(chosen_device)
            self.recorder.config.device = new_device_idx

            # Apply voice threshold
            try:
                new_threshold = float(voice_threshold_var.get())
                if new_threshold > 0:
                    self.voice_threshold_var.set(str(new_threshold))
                    self.asr_defaults["voice_threshold"] = new_threshold
                    self.continuous_listener.voice_threshold = new_threshold
            except ValueError:
                pass

            before = self.system_wide_input_var.get()
            after = system_wide_var.get()
            self.system_wide_input_var.set(after)
            if before != after:
                self._toggle_system_wide_input()
            self.status_var.set("Properties updated")
            self.dict_reading_entry = None
            self.dict_surface_entry = None
            self.dict_list = None
            self.properties_window = None
            win.destroy()

        buttons = tk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(12, 8), before=dict_frame)
        tk.Button(buttons, text="Apply", command=apply_and_close, width=10).pack(side=tk.LEFT)
        tk.Button(buttons, text="Cancel", command=win.destroy, width=10).pack(side=tk.RIGHT)

        def on_close() -> None:
            self.dict_reading_entry = None
            self.dict_surface_entry = None
            self.dict_list = None
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

    def _apply_asr_settings(self) -> None:
        whisper_model_name = (
            self.whisper_model_name_var.get().strip() or "Qwen/Qwen3-ASR-0.6B"
        )
        whisper_device = self.whisper_device_var.get().strip() or "auto"
        whisper_compute_type = self.whisper_compute_type_var.get().strip() or "int8"
        whisper_download_dir = self.root_dir / str(self.asr_defaults.get("whisper_download_dir", "models/whisper"))

        self.asr_defaults["whisper_model_name"] = whisper_model_name
        self.asr_defaults["whisper_device"] = whisper_device
        self.asr_defaults["whisper_compute_type"] = whisper_compute_type
        self.asr_defaults["whisper_download_dir"] = str(whisper_download_dir)

        self.asr_engine.configure(
            whisper_model_name=whisper_model_name,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            whisper_download_dir=whisper_download_dir,
        )

    def _download_asr_model_clicked(self, model_name: str, device: str, compute_type: str) -> None:
        self.status_var.set("Downloading ASR model...")
        threading.Thread(
            target=self._download_asr_model_worker,
            args=(model_name, device, compute_type),
            daemon=True,
        ).start()

    def _download_asr_model_worker(self, model_name: str, device: str, compute_type: str) -> None:
        result: dict[str, str] = {"model_path": "", "error": ""}
        try:
            whisper_download_dir = self.root_dir / str(
                self.asr_defaults.get("whisper_download_dir", "models/whisper")
            )
            self.asr_engine.configure(
                whisper_model_name=model_name,
                whisper_device=device,
                whisper_compute_type=compute_type,
                whisper_download_dir=whisper_download_dir,
            )
            target_dir = self.asr_engine.get_whisper_download_target_dir(model_name=model_name)

            def run_download() -> None:
                try:
                    result["model_path"] = self.asr_engine.download_whisper_model(model_name=model_name)
                except Exception as exc:  # noqa: BLE001
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        self.logger.exception("ASR model download failed")
                    except Exception:
                        pass

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
                        "Downloading ASR model... "
                        f"{self._format_size(downloaded)} downloaded "
                        f"({self._format_elapsed(elapsed_s)})"
                    ),
                )
                time.sleep(1.0)
            download_thread.join()
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("ASR model download failed")
            result["error"] = str(exc)
        self.root.after(0, self._on_download_asr_model_done, result["model_path"], result["error"])

    def _on_download_asr_model_done(self, model_path: str, error: str) -> None:
        if error:
            self.status_var.set("ASR model download failed")
            messagebox.showerror(
                "ASR model download error",
                f"{self._format_download_error(error)}\n\nLog: {self.root_dir / 'logs' / 'app.log'}",
            )
            return
        self.status_var.set("ASR model ready")
        messagebox.showinfo("ASR model", f"Model is ready at:\n{model_path}")

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
        if "model_not_found_and_auto_download_disabled" in raw:
            return (
                "Model was not found locally and auto-download is disabled.\n"
                "Use 'Download LLM Model' or enable auto-download in settings."
            )
        if "whisper_model_download_failed" in raw or "model_download_failed" in raw:
            return (
                "Model download failed.\n"
                "Please check network/proxy/firewall settings and try again."
            )
        if "qwen_asr_not_installed" in raw:
            return (
                "Qwen ASR backend (qwen-asr) is missing.\n"
                "Install dependencies with 'pip install -r requirements.txt'."
            )
        if "torch_not_installed" in raw:
            return (
                "PyTorch is missing for Qwen ASR backend.\n"
                "Install dependencies with 'pip install -r requirements.txt'."
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

    def _refresh_dictionary_list(self) -> None:
        if self.dict_list is None or not self.dict_list.winfo_exists():
            return
        self.dict_entries = self.personal_dictionary.list_entries()
        self.dict_list.delete(0, tk.END)
        for item in self.dict_entries:
            self.dict_list.insert(tk.END, f"{item.reading} -> {item.surface} ({item.count})")

    def _on_dictionary_selected(self, event):  # noqa: ANN001
        if self.dict_list is None or self.dict_reading_entry is None or self.dict_surface_entry is None:
            return
        if not self.dict_list.curselection():
            return
        idx = self.dict_list.curselection()[0]
        item = self.dict_entries[idx]
        self.dict_reading_entry.delete(0, tk.END)
        self.dict_reading_entry.insert(0, item.reading)
        self.dict_surface_entry.delete(0, tk.END)
        self.dict_surface_entry.insert(0, item.surface)

    def _add_dictionary_entry(self) -> None:
        if self.dict_reading_entry is None or self.dict_surface_entry is None:
            return
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
        if self.dict_reading_entry is None:
            return
        reading = self.dict_reading_entry.get().strip()
        if not reading:
            messagebox.showwarning("No target", "Please select a reading to remove.")
            return
        self.personal_dictionary.remove(reading)
        self._refresh_dictionary_list()
        self.status_var.set("Dictionary removed")

    # ------------------------------------------------------------------
    # Continuous (real-time VAD) mode
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
        self.status_var.set("Ready (Ctrl+Space / Ctrl+Shift+Space)")

    def _on_continuous_utterance(self, audio_data) -> None:  # noqa: ANN001
        """Called from ContinuousListener background thread when an utterance ends."""
        if not self._processing_lock.acquire(blocking=False):
            # Already processing a previous utterance; skip this one
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

        self.record_button.config(text="Start Recording", bg="#1f6feb", activebackground="#2f81f7", state=tk.DISABLED)
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
            self.root.after(
                0,
                self._apply_results,
                raw_asr,
                final,
                "",
                llm_result.fallback_reason,
                timings,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            self.root.after(0, self._apply_results, "", "", str(exc), "", timings)

    def _execute_voice_command(self, cmd: VoiceCommand) -> None:
        self._stop_processing_indicator()
        if not self._continuous_mode_active:
            self.record_button.config(state=tk.NORMAL)

        if cmd.action == "dict_add":
            reading = cmd.args.get("reading", "").strip()
            surface = cmd.args.get("surface", "").strip()
            try:
                self.personal_dictionary.add_or_update(reading=reading, surface=surface)
                self._refresh_dictionary_list()
                self.status_var.set(f"辞書登録: {reading} → {surface}")
            except ValueError as exc:
                self.status_var.set(f"辞書登録エラー: {exc}")

        elif cmd.action == "dict_remove":
            reading = cmd.args.get("reading", "").strip()
            self.personal_dictionary.remove(reading)
            self._refresh_dictionary_list()
            self.status_var.set(f"辞書削除: {reading}")

        elif cmd.action == "clear":
            self._set_text(self.final_text, "")
            if self.asr_text is not None:
                self._set_text(self.asr_text, "")
            self.current_raw_text = ""
            self.status_var.set("テキストをクリアしました")

        elif cmd.action == "copy":
            text = self.final_text.get("1.0", tk.END).strip()
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.status_var.set("テキストをコピーしました")
            else:
                self.status_var.set("コピー対象のテキストがありません")

        elif cmd.action == "properties":
            self._open_properties_dialog()
            self.status_var.set("プロパティを開きました")

        elif cmd.action == "launch_any":
            query = cmd.args.get("query", "")
            threading.Thread(target=self._launch_app_worker, args=(query,), daemon=True).start()

        elif cmd.action == "close_app":
            query = cmd.args.get("query", "")
            threading.Thread(target=self._close_app_worker, args=(query,), daemon=True).start()

    def _launch_app_worker(self, query: str) -> None:
        try:
            label = self.app_launcher.launch(query)
            self.root.after(0, self.status_var.set, f"{label} を起動しました")
        except ValueError:
            self.root.after(0, self.status_var.set, f"アプリが見つかりません: {query}")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to launch app: %s", query)
            self.root.after(0, self.status_var.set, f"起動エラー: {exc}")

    def _close_app_worker(self, query: str) -> None:
        try:
            label = self.app_launcher.close(query)
            self.root.after(0, self.status_var.set, f"{label} を終了しました")
        except ValueError:
            self.root.after(0, self.status_var.set, f"アプリが見つかりません: {query}")
        except RuntimeError as exc:
            self.root.after(0, self.status_var.set, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to close app: %s", query)
            self.root.after(0, self.status_var.set, f"終了エラー: {exc}")

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

    def _on_close(self) -> None:
        self.continuous_listener.stop()
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
                "Try a shorter recording and switch ASR device (auto/cpu) in Properties."
            )
        if "qwen_asr_not_installed" in normalized:
            return "Qwen ASR backend is not installed. Run: pip install -r requirements.txt"
        if "torch_not_installed" in normalized:
            return "PyTorch is not installed. Run: pip install -r requirements.txt"
        if "vector too long" in raw.lower():
            return (
                "Audio segment is too long for one-pass transcription.\n"
                "Please try a shorter recording segment and retry."
            )
        return raw or "Unknown error"


def build_app(
    root: tk.Tk,
    root_dir: Path,
    audio_config: AudioConfig,
    storage: Storage,
    rules: dict,
    personal_dictionary: PersonalDictionary,
    enable_system_wide_input_default: bool,
    llm_defaults: dict,
    asr_defaults: dict,
) -> VoiceInputApp:
    engine = ASREngine(
        sample_rate_hz=audio_config.sample_rate_hz,
        whisper_model_name=str(asr_defaults.get("whisper_model_name", "Qwen/Qwen3-ASR-0.6B")),
        whisper_device=str(asr_defaults.get("whisper_device", "auto")),
        whisper_compute_type=str(asr_defaults.get("whisper_compute_type", "int8")),
        whisper_download_dir=root_dir / str(asr_defaults.get("whisper_download_dir", "models/whisper")),
    )
    saved_device_str = asr_defaults.get("audio_input_device", None)
    audio_config.device = _parse_device_choice(str(saved_device_str)) if saved_device_str else None
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
        asr_defaults=asr_defaults,
        root_dir=root_dir,
        enable_system_wide_input_default=enable_system_wide_input_default,
    )
