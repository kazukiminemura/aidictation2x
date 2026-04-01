from src.system_wide_input import SystemWideInput


class _DummyListener:
    def __init__(self, hotkeys) -> None:
        self.hotkeys = hotkeys
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _DummyController:
    pass


def test_start_registers_record_and_continuous_hotkeys(monkeypatch) -> None:
    created: list[_DummyListener] = []

    def fake_global_hotkeys(hotkeys):
        listener = _DummyListener(hotkeys)
        created.append(listener)
        return listener

    monkeypatch.setattr("src.system_wide_input.keyboard.GlobalHotKeys", fake_global_hotkeys)
    monkeypatch.setattr("src.system_wide_input.keyboard.Controller", lambda: _DummyController())

    swi = SystemWideInput(
        dispatch_on_ui=lambda cb: cb(),
        on_toggle=lambda: None,
        on_toggle_continuous=lambda: None,
    )

    swi.start()

    assert len(created) == 1
    assert "<ctrl>+<shift>+<space>" in created[0].hotkeys
    assert "<ctrl>+<alt>+<space>" in created[0].hotkeys
    assert created[0].started is True


def test_continuous_hotkey_dispatches_callback(monkeypatch) -> None:
    monkeypatch.setattr("src.system_wide_input.keyboard.Controller", lambda: _DummyController())

    dispatched: list[str] = []

    def dispatch_on_ui(callback) -> None:
        dispatched.append("dispatch")
        callback()

    swi = SystemWideInput(
        dispatch_on_ui=dispatch_on_ui,
        on_toggle=lambda: dispatched.append("record"),
        on_toggle_continuous=lambda: dispatched.append("continuous"),
    )

    swi._on_continuous_hotkey()

    assert dispatched == ["dispatch", "continuous"]
