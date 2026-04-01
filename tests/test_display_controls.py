from src.display_controls import DisplayController


def test_set_brightness_clamps_value(monkeypatch) -> None:
    controller = DisplayController()
    calls: list[int] = []
    monkeypatch.setattr("src.display_controls.sbc.set_brightness", lambda value: calls.append(value))

    actual = controller.set_brightness(120)

    assert actual == 100
    assert calls == [100]


def test_change_brightness_uses_current_value(monkeypatch) -> None:
    controller = DisplayController()
    monkeypatch.setattr(controller, "get_brightness", lambda: 35)
    monkeypatch.setattr(controller, "set_brightness", lambda value: value)

    actual = controller.change_brightness(10)

    assert actual == 45


def test_get_brightness_reads_first_detected_value(monkeypatch) -> None:
    controller = DisplayController()
    monkeypatch.setattr("src.display_controls.sbc.get_brightness", lambda: [62, 40])

    actual = controller.get_brightness()

    assert actual == 62


def test_toggle_night_light_from_disabled(monkeypatch) -> None:
    controller = DisplayController()
    writes: list[bytes] = []
    source = bytes([0] * 43)

    monkeypatch.setattr(controller, "_read_binary_value", lambda _path: source)
    monkeypatch.setattr(controller, "_write_binary_value", lambda _path, value: writes.append(value))

    enabled = controller.toggle_night_light()

    assert enabled is True
    assert writes
    assert writes[0][18] == 0x15
    assert writes[0][23] == 0x10
    assert writes[0][24] == 0x00


def test_set_night_light_strength_updates_temperature_bytes(monkeypatch) -> None:
    controller = DisplayController()
    writes: list[bytes] = []
    source = bytearray(0x25)

    monkeypatch.setattr(controller, "_read_binary_value", lambda _path: bytes(source))
    monkeypatch.setattr(controller, "_write_binary_value", lambda _path, value: writes.append(value))

    actual = controller.set_night_light_strength(75)

    assert actual == 75
    assert writes
    assert writes[0][0x23] != 0
    assert writes[0][0x24] != 0
