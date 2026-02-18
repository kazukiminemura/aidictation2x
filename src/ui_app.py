import logging
import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import messagebox
from tkinter import ttk

from .asr import ASREngine
from .autonomous_agent import AutonomousAgentResult, ExternalAPIAutonomousAgent, InternalAutonomousAgent
from .audio_capture import AudioConfig, AudioRecorder
from .business_email import to_business_email
from .llm_post_editor import LLMOptions, LLMPostEditor
from .personal_dictionary import PersonalDictionary
from .storage import Storage
from .system_wide_input import SystemWideInput
from .text_processing import ProcessOptions, process_text

ASR_MODEL_CHOICES = (
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B",
)


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
        self.external_agent_enabled_var = tk.BooleanVar(
            value=bool(self.llm_defaults.get("external_agent_enabled", False))
        )
        self.external_agent_url_var = tk.StringVar(
            value=str(self.llm_defaults.get("external_agent_url", "http://127.0.0.1:8000/v1/agent/chat"))
        )
        self.whisper_model_name_var = tk.StringVar(
            value=str(self.asr_defaults.get("whisper_model_name", "Qwen/Qwen3-ASR-0.6B"))
        )
        self.whisper_device_var = tk.StringVar(value=str(self.asr_defaults.get("whisper_device", "auto")))
        self.whisper_compute_type_var = tk.StringVar(
            value=str(self.asr_defaults.get("whisper_compute_type", "int8"))
        )
        self.properties_window: tk.Toplevel | None = None
        self.agent_response_text: tk.Text | None = None
        self.rest_response_text: tk.Text | None = None
        self.agent_goal_var = tk.StringVar(value="")
        self.agent_run_button: tk.Button | None = None
        self._agent_running = False
        self.autonomous_agent_mode_var = tk.StringVar(
            value=str(self.llm_defaults.get("autonomous_agent_mode", "internal"))
        )
        self.autonomous_agent_external_url_var = tk.StringVar(
            value=str(self.llm_defaults.get("autonomous_agent_external_url", "http://127.0.0.1:8000/v1/agent/run"))
        )
        self.asr_text: tk.Text | None = None
        self.dict_reading_entry: tk.Entry | None = None
        self.dict_surface_entry: tk.Entry | None = None
        self.dict_list: tk.Listbox | None = None
        self.dict_entries = []
        self._processing_active = False
        self._processing_started = 0.0
        self._processing_phase = "Processing"
        self._processing_tick_token = 0

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
        agent_tab = tk.Frame(tabs, bg="#0a0e14")
        rest_tab = tk.Frame(tabs, bg="#0a0e14")
        tabs.add(asr_tab, text="ASR Text")
        tabs.add(final_tab, text="Final")
        tabs.add(agent_tab, text="AI Agent")
        tabs.add(rest_tab, text="REST Raw")
        tab_selected_colors = {
            0: "#14532d",  # ASR Text
            1: "#1d4ed8",  # Final
            2: "#7c2d12",  # AI Agent
            3: "#4c1d95",  # REST Raw
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

        agent_controls = tk.Frame(agent_tab, bg="#0a0e14")
        agent_controls.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            agent_controls,
            text="Goal",
            fg="#8b9fb6",
            bg="#0a0e14",
            font=("Consolas", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Entry(
            agent_controls,
            textvariable=self.agent_goal_var,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.agent_run_button = tk.Button(
            agent_controls,
            text="Run Agent",
            command=self._run_autonomous_agent_clicked,
            bg="#7c2d12",
            fg="#ffffff",
            activebackground="#9a3412",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            font=("Consolas", 9, "bold"),
            cursor="hand2",
        )
        self.agent_run_button.pack(side=tk.LEFT)

        self.agent_response_text = tk.Text(
            agent_tab,
            height=18,
            wrap=tk.WORD,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.agent_response_text.pack(fill=tk.BOTH, expand=True)

        self.rest_response_text = tk.Text(
            rest_tab,
            height=18,
            wrap=tk.WORD,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.rest_response_text.pack(fill=tk.BOTH, expand=True)

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
        external_agent_enabled_var = tk.BooleanVar(value=self.external_agent_enabled_var.get())
        external_agent_url_var = tk.StringVar(value=self.external_agent_url_var.get())
        autonomous_agent_mode_var = tk.StringVar(value=self.autonomous_agent_mode_var.get())
        autonomous_agent_external_url_var = tk.StringVar(value=self.autonomous_agent_external_url_var.get())
        whisper_model_name_var = tk.StringVar(value=self.whisper_model_name_var.get())
        whisper_device_var = tk.StringVar(value=self.whisper_device_var.get())
        whisper_compute_type_var = tk.StringVar(value=self.whisper_compute_type_var.get())

        frame = tk.Frame(win, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Checkbutton(frame, text="Auto edit", variable=auto_edit_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Remove fillers", variable=remove_fillers_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Remove habits", variable=remove_habits_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Convert to business email", variable=business_email_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Enable LLM correction", variable=llm_enabled_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Use external AI agent", variable=external_agent_enabled_var).pack(
            anchor=tk.W, pady=4
        )
        tk.Label(frame, text="External agent URL").pack(anchor=tk.W, pady=(8, 0))
        tk.Entry(frame, textvariable=external_agent_url_var).pack(anchor=tk.W, fill=tk.X)
        tk.Label(frame, text="Autonomous agent mode").pack(anchor=tk.W, pady=(8, 0))
        tk.OptionMenu(frame, autonomous_agent_mode_var, "internal", "external_api").pack(anchor=tk.W, fill=tk.X)
        tk.Label(frame, text="Autonomous external API URL").pack(anchor=tk.W, pady=(8, 0))
        tk.Entry(frame, textvariable=autonomous_agent_external_url_var).pack(anchor=tk.W, fill=tk.X)
        tk.Checkbutton(
            frame,
            text="System-wide input (paste to active app on completion)",
            variable=system_wide_var,
        ).pack(anchor=tk.W, pady=4)
        tk.Label(frame, text="ASR model name").pack(anchor=tk.W, pady=(8, 0))
        ttk.Combobox(
            frame,
            textvariable=whisper_model_name_var,
            values=ASR_MODEL_CHOICES,
            state="normal",
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(frame, text="ASR device").pack(anchor=tk.W, pady=(8, 0))
        tk.OptionMenu(frame, whisper_device_var, "auto", "cpu", "cuda").pack(anchor=tk.W, fill=tk.X)
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
            self.external_agent_enabled_var.set(external_agent_enabled_var.get())
            self.llm_defaults["external_agent_enabled"] = bool(external_agent_enabled_var.get())
            self.external_agent_url_var.set(
                external_agent_url_var.get().strip() or "http://127.0.0.1:8000/v1/agent/chat"
            )
            self.llm_defaults["external_agent_url"] = self.external_agent_url_var.get()
            self.autonomous_agent_mode_var.set(
                autonomous_agent_mode_var.get().strip() or "internal"
            )
            self.llm_defaults["autonomous_agent_mode"] = self.autonomous_agent_mode_var.get()
            self.autonomous_agent_external_url_var.set(
                autonomous_agent_external_url_var.get().strip() or "http://127.0.0.1:8000/v1/agent/run"
            )
            self.llm_defaults["autonomous_agent_external_url"] = self.autonomous_agent_external_url_var.get()
            self.whisper_model_name_var.set(
                whisper_model_name_var.get().strip() or "Qwen/Qwen3-ASR-0.6B"
            )
            self.whisper_device_var.set(whisper_device_var.get())
            self.whisper_compute_type_var.set(whisper_compute_type_var.get())
            self._apply_asr_settings()

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

    def _run_autonomous_agent_clicked(self) -> None:
        if self._agent_running:
            return
        goal = self.agent_goal_var.get().strip()
        if not goal:
            messagebox.showwarning("Goal missing", "Please input an autonomous-agent goal.")
            return
        self._agent_running = True
        if self.agent_run_button is not None:
            self.agent_run_button.config(state=tk.DISABLED)
        self.status_var.set("Autonomous agent running...")
        mode = (self.autonomous_agent_mode_var.get() or "internal").strip()
        endpoint = (self.autonomous_agent_external_url_var.get() or "").strip()
        threading.Thread(
            target=self._run_autonomous_agent_worker,
            args=(goal, mode, endpoint),
            daemon=True,
        ).start()

    def _run_autonomous_agent_worker(self, goal: str, mode: str, endpoint: str) -> None:
        try:
            if mode == "external_api":
                agent = ExternalAPIAutonomousAgent(endpoint_url=endpoint or "http://127.0.0.1:8000/v1/agent/run")
                result = agent.run(goal=goal, workspace_root=self.root_dir)
            else:
                agent = InternalAutonomousAgent(workspace_root=self.root_dir)
                result = agent.run(goal=goal)
            self.root.after(0, self._on_autonomous_agent_done, result, "")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Autonomous agent failed")
            self.root.after(0, self._on_autonomous_agent_done, None, str(exc))

    def _on_autonomous_agent_done(self, result: AutonomousAgentResult | None, error: str) -> None:
        self._agent_running = False
        if self.agent_run_button is not None:
            self.agent_run_button.config(state=tk.NORMAL)

        if error:
            self.status_var.set("Autonomous agent failed")
            messagebox.showerror("Autonomous agent error", error)
            return
        if result is None:
            self.status_var.set("Autonomous agent failed")
            messagebox.showerror("Autonomous agent error", "Unknown error")
            return

        self.status_var.set(f"Autonomous agent done ({result.mode})")
        if self.agent_response_text is not None:
            self._set_text(self.agent_response_text, self._format_agent_result(result))
        if self.rest_response_text is not None:
            self._set_text(self.rest_response_text, result.external_raw_response or "")

    @staticmethod
    def _format_agent_result(result: AutonomousAgentResult) -> str:
        lines = [
            f"Goal: {result.goal}",
            f"Mode: {result.mode}",
            f"Success: {result.success}",
            f"Summary: {result.summary}",
            "",
            "Steps:",
        ]
        for step in result.steps:
            output = f" -> {step.output_path}" if step.output_path else ""
            detail = f" ({step.detail})" if step.detail else ""
            lines.append(f"- {step.name}: {step.status}{detail}{output}")
        if result.artifact_paths:
            lines.append("")
            lines.append("Artifacts:")
            for path in result.artifact_paths:
                lines.append(f"- {path}")
        if result.report_path:
            lines.append("")
            lines.append(f"Report: {result.report_path}")
        return "\n".join(lines)

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
                    enabled=bool(self.llm_enabled_var.get() or self.external_agent_enabled_var.get()),
                    strength=str(self.llm_defaults.get("strength", "medium")),
                    max_input_chars=int(self.llm_defaults.get("max_input_chars", 1200)),
                    max_change_ratio=float(self.llm_defaults.get("max_change_ratio", 0.35)),
                    domain_hint=str(self.llm_defaults.get("domain_hint", "")),
                    external_agent_enabled=bool(self.external_agent_enabled_var.get()),
                    external_agent_url=str(self.external_agent_url_var.get()).strip()
                    or "http://127.0.0.1:8000/v1/agent/chat",
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
                bool(self.external_agent_enabled_var.get()),
                llm_result.external_agent_response,
                llm_result.external_agent_raw_response,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            self.root.after(0, self._apply_results, "", "", str(exc), "", timings)

    def _apply_results(
        self,
        asr_text_value: str,
        final: str,
        error: str,
        fallback_reason: str = "",
        timings: dict[str, int] | None = None,
        external_agent_used: bool = False,
        external_agent_response: str = "",
        external_agent_raw_response: str = "",
    ) -> None:
        self._stop_processing_indicator()
        self.record_button.config(state=tk.NORMAL)

        if error:
            self.status_var.set("Error")
            messagebox.showerror("Processing error", self._format_processing_error(error))
            return

        timing_suffix = self._format_timing_suffix(timings)
        self._set_text(self.final_text, final)
        if self.asr_text is not None:
            self._set_text(self.asr_text, asr_text_value)
        if self.agent_response_text is not None:
            self._set_text(
                self.agent_response_text,
                external_agent_response if external_agent_used else "",
            )
        if self.rest_response_text is not None:
            self._set_text(
                self.rest_response_text,
                external_agent_raw_response if external_agent_used else "",
            )
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
