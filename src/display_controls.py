"""Windows display controls for brightness and Night Light."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

import screen_brightness_control as sbc

try:
    import winreg
except ImportError:  # pragma: no cover - Windows only feature
    winreg = None  # type: ignore[assignment]


_STATE_KEY_PATH = (
    r"Software\Microsoft\Windows\CurrentVersion\CloudStore\Store\DefaultAccount\Current"
    r"\default$windows.data.bluelightreduction.bluelightreductionstate"
    r"\windows.data.bluelightreduction.bluelightreductionstate"
)
_SETTINGS_KEY_PATH = (
    r"Software\Microsoft\Windows\CurrentVersion\CloudStore\Store\DefaultAccount\Current"
    r"\default$windows.data.bluelightreduction.settings"
    r"\windows.data.bluelightreduction.settings"
)
_MIN_NIGHT_LIGHT_KELVIN = 1200
_MAX_NIGHT_LIGHT_KELVIN = 6500


@dataclass(frozen=True)
class DisplayState:
    brightness: int | None
    night_light_enabled: bool | None
    night_light_strength: int | None


class DisplayController:
    """Best-effort Windows display controls."""

    def get_state(self) -> DisplayState:
        brightness = None
        try:
            brightness = self.get_brightness()
        except RuntimeError:
            pass

        night_light_enabled = None
        night_light_strength = None
        if self.is_night_light_supported():
            night_light_enabled = self.is_night_light_enabled()
            night_light_strength = self.get_night_light_strength()

        return DisplayState(
            brightness=brightness,
            night_light_enabled=night_light_enabled,
            night_light_strength=night_light_strength,
        )

    def get_brightness(self) -> int:
        try:
            values = [value for value in sbc.get_brightness() if value is not None]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("明るさを取得できませんでした") from exc
        if not values:
            raise RuntimeError("明るさを取得できませんでした")
        return self._clamp_percentage(int(values[0]))

    def set_brightness(self, value: int) -> int:
        level = self._clamp_percentage(value)
        try:
            sbc.set_brightness(level)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("明るさの変更に対応していないディスプレイです") from exc
        return level

    def change_brightness(self, delta: int) -> int:
        current = self.get_brightness()
        return self.set_brightness(current + delta)

    def is_night_light_supported(self) -> bool:
        if winreg is None:
            return False
        return self._registry_value_exists(_STATE_KEY_PATH) and self._registry_value_exists(_SETTINGS_KEY_PATH)

    def is_night_light_enabled(self) -> bool:
        data = self._read_binary_value(_STATE_KEY_PATH)
        if len(data) <= 18:
            raise RuntimeError("夜間モードの状態を取得できませんでした")
        return data[18] == 0x15

    def enable_night_light(self) -> bool:
        if not self.is_night_light_enabled():
            return self.toggle_night_light()
        return True

    def disable_night_light(self) -> bool:
        if self.is_night_light_enabled():
            return self.toggle_night_light()
        return False

    def toggle_night_light(self) -> bool:
        data = bytearray(self._read_binary_value(_STATE_KEY_PATH))
        if len(data) <= 18:
            raise RuntimeError("夜間モードの状態を更新できませんでした")

        enabled = data[18] == 0x15
        if enabled:
            new_data = bytearray(41)
            new_data[0:22] = data[0:22]
            tail = data[25:43]
            new_data[23 : 23 + len(tail)] = tail
            new_data[18] = 0x13
        else:
            new_data = bytearray(43)
            new_data[0:22] = data[0:22]
            tail = data[23:41]
            new_data[25 : 25 + len(tail)] = tail
            new_data[18] = 0x15
            new_data[23] = 0x10
            new_data[24] = 0x00

        self._bump_version_bytes(new_data)
        self._write_binary_value(_STATE_KEY_PATH, bytes(new_data))
        return not enabled

    def get_night_light_strength(self) -> int:
        data = self._read_binary_value(_SETTINGS_KEY_PATH)
        if len(data) <= 0x24:
            raise RuntimeError("夜間モードの強さを取得できませんでした")
        kelvin = (data[0x24] * 64) + ((data[0x23] - 128) / 2)
        percentage = 100 - ((kelvin - _MIN_NIGHT_LIGHT_KELVIN) / (_MAX_NIGHT_LIGHT_KELVIN - _MIN_NIGHT_LIGHT_KELVIN)) * 100
        return self._clamp_percentage(round(percentage))

    def set_night_light_strength(self, value: int) -> int:
        data = bytearray(self._read_binary_value(_SETTINGS_KEY_PATH))
        if len(data) <= 0x24:
            raise RuntimeError("夜間モードの強さを更新できませんでした")

        percentage = self._clamp_percentage(value)
        kelvin = round(
            _MAX_NIGHT_LIGHT_KELVIN
            - (percentage / 100) * (_MAX_NIGHT_LIGHT_KELVIN - _MIN_NIGHT_LIGHT_KELVIN)
        )
        temp_hi = int(kelvin // 64)
        temp_lo = int(((kelvin - (temp_hi * 64)) * 2) + 128)

        data[0x23] = temp_lo
        data[0x24] = temp_hi
        self._bump_version_bytes(data)
        self._write_binary_value(_SETTINGS_KEY_PATH, bytes(data))
        return percentage

    def _run_powershell(self, script: str) -> str:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or "PowerShell failed"
            raise RuntimeError(error)
        return result.stdout.strip()

    @staticmethod
    def _clamp_percentage(value: int) -> int:
        return max(0, min(100, int(value)))

    @staticmethod
    def _bump_version_bytes(data: bytearray) -> None:
        for index in range(10, min(15, len(data))):
            if data[index] != 0xFF:
                data[index] = (data[index] + 1) % 256
                return

    def _registry_value_exists(self, path: str) -> bool:
        if winreg is None:
            return False
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ) as key:
                winreg.QueryValueEx(key, "Data")
            return True
        except OSError:
            return False

    def _read_binary_value(self, path: str) -> bytes:
        if winreg is None:
            raise RuntimeError("Windows のみ対応しています")
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ) as key:
                data, reg_type = winreg.QueryValueEx(key, "Data")
        except OSError as exc:
            raise RuntimeError("夜間モード機能を利用できませんでした") from exc
        if reg_type != winreg.REG_BINARY:
            raise RuntimeError("夜間モードの設定形式が不正です")
        return bytes(data)

    def _write_binary_value(self, path: str, value: bytes) -> None:
        if winreg is None:
            raise RuntimeError("Windows のみ対応しています")
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "Data", 0, winreg.REG_BINARY, value)
        except OSError as exc:
            raise RuntimeError("夜間モードの設定更新に失敗しました") from exc
