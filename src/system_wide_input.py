import time
from typing import Callable

import pyperclip
from pynput import keyboard


class SystemWideInput:
    def __init__(
        self,
        dispatch_on_ui: Callable[[Callable[[], None]], None],
        on_toggle: Callable[[], None],
        on_toggle_continuous: Callable[[], None] | None = None,
    ):
        self.dispatch_on_ui = dispatch_on_ui
        self.on_toggle = on_toggle
        self.on_toggle_continuous = on_toggle_continuous
        self._listener: keyboard.GlobalHotKeys | None = None
        self._controller = keyboard.Controller()

    def start(self) -> None:
        if self._listener is not None:
            return
        hotkeys = {
            "<ctrl>+<shift>+<space>": self._on_hotkey,
        }
        if self.on_toggle_continuous is not None:
            hotkeys["<ctrl>+<alt>+<space>"] = self._on_continuous_hotkey
        self._listener = keyboard.GlobalHotKeys(hotkeys)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None

    def _on_hotkey(self) -> None:
        self.dispatch_on_ui(self.on_toggle)

    def _on_continuous_hotkey(self) -> None:
        if self.on_toggle_continuous is None:
            return
        self.dispatch_on_ui(self.on_toggle_continuous)

    def paste_to_active_app(self, text: str) -> None:
        if not text:
            return
        pyperclip.copy(text)
        time.sleep(0.05)
        with self._controller.pressed(keyboard.Key.ctrl):
            self._controller.press("v")
            self._controller.release("v")
