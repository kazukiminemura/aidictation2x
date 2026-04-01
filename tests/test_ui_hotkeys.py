from types import SimpleNamespace

from src.ui_app import VoiceInputApp


class _FakeVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _FakeButton:
    def __init__(self) -> None:
        self.state = None

    def config(self, **kwargs) -> None:  # noqa: ANN003
        self.state = kwargs.get("state", self.state)


class _FakeProgressbar:
    def __init__(self) -> None:
        self.started = False
        self.mode = None

    def config(self, **kwargs) -> None:  # noqa: ANN003
        self.mode = kwargs.get("mode", self.mode)

    def start(self, _interval: int) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


def test_space_hotkey_press_toggles_recording_without_alt_or_shift() -> None:
    calls: list[str] = []
    app = VoiceInputApp.__new__(VoiceInputApp)
    app.hotkey_pressed = False
    app.continuous_hotkey_pressed = False
    app.toggle_recording = lambda: calls.append("record")
    app._toggle_continuous = lambda: calls.append("continuous")

    result = VoiceInputApp._on_space_hotkey_press(app, SimpleNamespace(state=0x0004))

    assert result == "break"
    assert app.hotkey_pressed is True
    assert calls == ["record"]


def test_space_hotkey_press_toggles_continuous_with_alt() -> None:
    calls: list[str] = []
    app = VoiceInputApp.__new__(VoiceInputApp)
    app.hotkey_pressed = False
    app.continuous_hotkey_pressed = False
    app.toggle_recording = lambda: calls.append("record")
    app._toggle_continuous = lambda: calls.append("continuous")

    result = VoiceInputApp._on_space_hotkey_press(app, SimpleNamespace(state=0x000C))

    assert result == "break"
    assert app.continuous_hotkey_pressed is True
    assert calls == ["continuous"]


def test_asr_download_progress_helpers_toggle_widgets() -> None:
    app = VoiceInputApp.__new__(VoiceInputApp)
    app._asr_download_in_progress = False
    app.asr_download_progress_var = _FakeVar()
    app.status_var = _FakeVar()
    app.asr_download_button = _FakeButton()
    app.asr_download_progressbar = _FakeProgressbar()

    VoiceInputApp._start_asr_download_progress(app, "Qwen/Qwen3-ASR-1.7B", "gpu")
    assert app._asr_download_in_progress is True
    assert app.asr_download_button.state == "disabled"
    assert app.asr_download_progressbar.started is True
    assert "Preparing Qwen/Qwen3-ASR-1.7B on GPU" in app.asr_download_progress_var.value

    VoiceInputApp._update_asr_download_progress(app, "Downloading files...")
    assert app.asr_download_progress_var.value == "Downloading files..."
    assert app.status_var.value == "Downloading files..."

    VoiceInputApp._stop_asr_download_progress(app, "ASR model ready")
    assert app._asr_download_in_progress is False
    assert app.asr_download_button.state == "normal"
    assert app.asr_download_progressbar.started is False
    assert app.asr_download_progress_var.value == "ASR model ready"
