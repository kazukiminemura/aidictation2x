from types import SimpleNamespace

from src.ui_app import VoiceInputApp


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
