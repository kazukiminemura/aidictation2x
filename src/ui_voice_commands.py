"""VoiceCommandsMixin: voice command dispatching and action worker threads.

Mixed into VoiceInputApp. Accesses self.* attributes set in VoiceInputApp.__init__.
"""
from __future__ import annotations

import threading
import tkinter as tk

from .voice_commands import VoiceCommand


class VoiceCommandsMixin:
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

        elif cmd.action == "browser_search":
            query = cmd.args.get("query", "")
            threading.Thread(target=self._browser_search_worker, args=(query,), daemon=True).start()

        elif cmd.action == "browser_open_result":
            query = cmd.args.get("query", "")
            threading.Thread(target=self._browser_open_result_worker, args=(query,), daemon=True).start()

        elif cmd.action == "browser_open_url":
            target = cmd.args.get("target", "")
            threading.Thread(target=self._browser_open_url_worker, args=(target,), daemon=True).start()

        elif cmd.action == "browser_back":
            threading.Thread(target=self._browser_back_worker, daemon=True).start()

        elif cmd.action == "browser_forward":
            threading.Thread(target=self._browser_forward_worker, daemon=True).start()

        elif cmd.action == "launch_any":
            query = cmd.args.get("query", "")
            threading.Thread(target=self._launch_app_worker, args=(query,), daemon=True).start()

        elif cmd.action == "close_app":
            query = cmd.args.get("query", "")
            threading.Thread(target=self._close_app_worker, args=(query,), daemon=True).start()

        elif cmd.action == "display_brightness_adjust":
            delta = int(cmd.args.get("delta", 0))
            threading.Thread(target=self._brightness_adjust_worker, args=(delta,), daemon=True).start()

        elif cmd.action == "display_brightness_set":
            level = int(cmd.args.get("level", 0))
            threading.Thread(target=self._brightness_set_worker, args=(level,), daemon=True).start()

        elif cmd.action == "night_light_toggle":
            threading.Thread(target=self._night_light_toggle_worker, daemon=True).start()

        elif cmd.action == "night_light_set_enabled":
            enabled = bool(cmd.args.get("enabled", False))
            threading.Thread(target=self._night_light_set_enabled_worker, args=(enabled,), daemon=True).start()

        elif cmd.action == "night_light_strength_set":
            strength = int(cmd.args.get("strength", 0))
            threading.Thread(target=self._night_light_strength_set_worker, args=(strength,), daemon=True).start()

    # ------------------------------------------------------------------
    # App launch / browser workers
    # ------------------------------------------------------------------

    def _launch_app_worker(self, query: str) -> None:
        try:
            label = self.app_launcher.launch(query)
            self.root.after(0, self.status_var.set, f"{label} を起動しました")
        except ValueError:
            self.root.after(0, self.status_var.set, f"アプリが見つかりません: {query}")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to launch app: %s", query)
            self.root.after(0, self.status_var.set, f"起動エラー: {exc}")

    def _browser_search_worker(self, query: str) -> None:
        try:
            label = self.app_launcher.browser_search(query)
            self.root.after(0, self.status_var.set, f"{label} で検索しました: {query}")
        except ValueError as exc:
            self.root.after(0, self.status_var.set, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to search in browser: %s", query)
            self.root.after(0, self.status_var.set, f"検索エラー: {exc}")

    def _browser_open_result_worker(self, query: str) -> None:
        try:
            label = self.app_launcher.browser_open_result(query)
            self.root.after(0, self.status_var.set, f"{label} で先頭候補を開きました: {query}")
        except ValueError as exc:
            self.root.after(0, self.status_var.set, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to open browser result: %s", query)
            self.root.after(0, self.status_var.set, f"リンク移動エラー: {exc}")

    def _browser_open_url_worker(self, target: str) -> None:
        try:
            label = self.app_launcher.browser_open_url(target)
            self.root.after(0, self.status_var.set, f"{label} で開きました: {target}")
        except ValueError as exc:
            self.root.after(0, self.status_var.set, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to open browser url: %s", target)
            self.root.after(0, self.status_var.set, f"リンク移動エラー: {exc}")

    def _browser_back_worker(self) -> None:
        try:
            label = self.app_launcher.browser_back()
            self.root.after(0, self.status_var.set, f"{label} で戻りました")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to navigate browser back")
            self.root.after(0, self.status_var.set, f"ブラウザ操作エラー: {exc}")

    def _browser_forward_worker(self) -> None:
        try:
            label = self.app_launcher.browser_forward()
            self.root.after(0, self.status_var.set, f"{label} で進みました")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to navigate browser forward")
            self.root.after(0, self.status_var.set, f"ブラウザ操作エラー: {exc}")

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

    # ------------------------------------------------------------------
    # Display control workers (called from voice commands)
    # ------------------------------------------------------------------

    def _brightness_adjust_worker(self, delta: int) -> None:
        try:
            level = self.display_controller.change_brightness(delta)
            self.root.after(0, self.status_var.set, f"明るさを {level}% にしました")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to adjust brightness")
            self.root.after(0, self.status_var.set, f"明るさ変更エラー: {exc}")

    def _brightness_set_worker(self, level: int) -> None:
        try:
            actual = self.display_controller.set_brightness(level)
            self.root.after(0, self.status_var.set, f"明るさを {actual}% にしました")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to set brightness")
            self.root.after(0, self.status_var.set, f"明るさ変更エラー: {exc}")

    def _night_light_toggle_worker(self) -> None:
        try:
            enabled = self.display_controller.toggle_night_light()
            label = "オン" if enabled else "オフ"
            self.root.after(0, self.status_var.set, f"夜間モードを {label} にしました")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to toggle Night Light")
            self.root.after(0, self.status_var.set, f"夜間モード変更エラー: {exc}")

    def _night_light_set_enabled_worker(self, enabled: bool) -> None:
        try:
            if enabled:
                self.display_controller.enable_night_light()
            else:
                self.display_controller.disable_night_light()
            label = "オン" if enabled else "オフ"
            self.root.after(0, self.status_var.set, f"夜間モードを {label} にしました")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to set Night Light enabled=%s", enabled)
            self.root.after(0, self.status_var.set, f"夜間モード変更エラー: {exc}")

    def _night_light_strength_set_worker(self, strength: int) -> None:
        try:
            actual = self.display_controller.set_night_light_strength(strength)
            self.root.after(0, self.status_var.set, f"夜間モードの強さを {actual}% にしました")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to set Night Light strength")
            self.root.after(0, self.status_var.set, f"夜間モード変更エラー: {exc}")
