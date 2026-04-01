"""VoiceInputApp: main application class (Mixin composition).

This module is the entry point for the UI layer. Heavy responsibilities are
split across focused mixins following SRP:

  RecordingMixin      -> ui_recording.py   (recording, pipeline, continuous)
  WorkersMixin        -> ui_workers.py     (model download, threshold calibration)
  VoiceCommandsMixin  -> ui_voice_commands.py (command dispatch and workers)
  PropertiesMixin     -> ui_properties.py  (properties dialog, dict, display)
  build_main_ui()     -> ui_builder.py     (widget construction)
"""
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .asr import ASREngine
from .audio_capture import AudioConfig, AudioRecorder
from .continuous_listener import ContinuousListener
from .app_launcher import AppLauncher
from .display_controls import DisplayController
from .llm_post_editor import LLMPostEditor
from .personal_dictionary import PersonalDictionary
from .storage import Storage
from .system_wide_input import SystemWideInput
from .ui_builder import build_main_ui, load_initial_state
from .ui_recording import RecordingMixin
from .ui_workers import WorkersMixin
from .ui_voice_commands import VoiceCommandsMixin
from .ui_properties import PropertiesMixin
from .ui_utils import parse_device_choice


class VoiceInputApp(RecordingMixin, WorkersMixin, VoiceCommandsMixin, PropertiesMixin):
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

        # UI state variables
        self.auto_edit_var = tk.BooleanVar(value=True)
        self.remove_fillers_var = tk.BooleanVar(value=True)
        self.remove_habits_var = tk.BooleanVar(value=True)
        self.business_email_var = tk.BooleanVar(value=False)
        self.system_wide_input_var = tk.BooleanVar(value=enable_system_wide_input_default)
        self.status_var = tk.StringVar(value="Starting...")
        self.current_raw_text = ""
        self.hotkey_pressed = False
        self.continuous_hotkey_pressed = False
        self.llm_enabled_var = tk.BooleanVar(value=bool(self.llm_defaults.get("enabled", False)))
        self.whisper_model_id_var = tk.StringVar(
            value=str(self.asr_defaults.get("whisper_model_id", "Qwen/Qwen3-ASR-1.7B"))
        )
        self.whisper_device_var = tk.StringVar(value=str(self.asr_defaults.get("whisper_device", "gpu")))
        _saved_device = self.asr_defaults.get("audio_input_device", None)
        self.audio_device_var = tk.StringVar(
            value=str(_saved_device) if _saved_device else "auto (system default)"
        )
        self.voice_threshold_var = tk.StringVar(
            value=str(self.asr_defaults.get("voice_threshold", "0.02"))
        )

        # Dialog widget references (set when properties window is open)
        self.properties_window: tk.Toplevel | None = None
        self.asr_download_button: tk.Button | None = None
        self.asr_download_progressbar: ttk.Progressbar | None = None
        self.asr_download_progress_var = tk.StringVar(value="")
        self._asr_download_in_progress = False
        self.asr_text: tk.Text | None = None
        self.dict_reading_entry: tk.Entry | None = None
        self.dict_surface_entry: tk.Entry | None = None
        self.dict_list: tk.Listbox | None = None
        self.dict_entries = []

        # Processing state
        self._processing_active = False
        self._processing_lock = threading.Lock()
        self._processing_started = 0.0
        self._processing_phase = "Processing"
        self._processing_tick_token = 0
        self._continuous_mode_active = False
        self.continuous_button: tk.Button | None = None

        # Services
        self.system_wide_input = SystemWideInput(
            dispatch_on_ui=lambda cb: self.root.after(0, cb),
            on_toggle=self.toggle_recording,
            on_toggle_continuous=self._toggle_continuous,
        )
        self.continuous_listener = ContinuousListener(
            config=self.recorder.config,
            on_utterance=self._on_continuous_utterance,
            on_voice_start=lambda: self.root.after(0, self.status_var.set, "Speaking..."),
            voice_threshold=float(self.voice_threshold_var.get()),
        )
        self.app_launcher = AppLauncher()
        self.display_controller = DisplayController()

        build_main_ui(self)
        self._bind_hotkeys()
        self._bind_context_menu()
        load_initial_state(self)
        self._refresh_dictionary_list()

        if self.system_wide_input_var.get():
            self.system_wide_input.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Hotkeys and context menu
    # ------------------------------------------------------------------

    def _bind_hotkeys(self) -> None:
        self.root.bind_all("<KeyPress-space>", self._on_space_hotkey_press)
        self.root.bind_all("<KeyRelease-space>", self._on_space_hotkey_release)

    def _bind_context_menu(self) -> None:
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Properties...", command=self._open_properties_dialog)
        self.root.bind("<Button-3>", self._show_context_menu)
        self.root.bind("<Control-Button-1>", self._show_context_menu)

    def _show_context_menu(self, event) -> None:  # noqa: ANN001
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _on_space_hotkey_press(self, event) -> None:  # noqa: ANN001
        ctrl_pressed = bool(event.state & 0x0004)
        shift_pressed = bool(event.state & 0x0001)
        alt_pressed = bool(event.state & 0x0008)
        if not ctrl_pressed:
            return
        if alt_pressed:
            if self.continuous_hotkey_pressed:
                return "break"
            self.continuous_hotkey_pressed = True
            self._toggle_continuous()
            return "break"
        if shift_pressed:
            return
        if self.hotkey_pressed:
            return "break"
        self.hotkey_pressed = True
        self.toggle_recording()
        return "break"

    def _on_space_hotkey_release(self, event) -> None:  # noqa: ANN001
        ctrl_pressed = bool(event.state & 0x0004)
        alt_pressed = bool(event.state & 0x0008)
        if ctrl_pressed and alt_pressed:
            self.continuous_hotkey_pressed = False
            return "break"
        if ctrl_pressed:
            self.hotkey_pressed = False
            return "break"

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _toggle_system_wide_input(self) -> None:
        if self.system_wide_input_var.get():
            self.system_wide_input.start()
            self.status_var.set("System-wide input: ON")
        else:
            self.system_wide_input.stop()
            self.status_var.set("System-wide input: OFF")

    def _apply_asr_settings(self) -> None:
        model_id = self.whisper_model_id_var.get().strip() or "Qwen/Qwen3-ASR-1.7B"
        device = self.whisper_device_var.get().strip() or "gpu"
        self.asr_defaults["whisper_model_id"] = model_id
        self.asr_defaults["whisper_device"] = device
        self.asr_engine.configure(device=device, model_id=model_id)

    def _on_close(self) -> None:
        self.continuous_listener.stop()
        self.system_wide_input.stop()
        self.root.destroy()


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
    whisper_download_dir = root_dir / str(asr_defaults.get("whisper_download_dir", "models/asr"))
    engine = ASREngine(
        sample_rate_hz=audio_config.sample_rate_hz,
        model_id=str(asr_defaults.get("whisper_model_id", "Qwen/Qwen3-ASR-1.7B")),
        device=str(asr_defaults.get("whisper_device", "gpu")),
        models_root_dir=whisper_download_dir,
        language=str(asr_defaults.get("asr_language", "ja")),
    )
    saved_device_str = asr_defaults.get("audio_input_device", None)
    audio_config.device = parse_device_choice(str(saved_device_str)) if saved_device_str else None
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
