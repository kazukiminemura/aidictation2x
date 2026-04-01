"""Shared UI helper functions for audio device selection."""
import sounddevice as sd


def get_input_device_choices() -> list[str]:
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


def parse_device_choice(value: str) -> "int | None":
    """Parse a device choice string back to an int index (or None for auto)."""
    v = (value or "").strip()
    if not v or v.startswith("auto"):
        return None
    try:
        return int(v.split(":")[0])
    except (ValueError, IndexError):
        return None
