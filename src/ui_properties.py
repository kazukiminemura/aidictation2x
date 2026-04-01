"""PropertiesMixin: properties dialog, dictionary UI, display control dialog.

Mixed into VoiceInputApp. Accesses self.* attributes set in VoiceInputApp.__init__.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .asr import get_supported_model_ids
from .ui_utils import get_input_device_choices, parse_device_choice


class PropertiesMixin:
    def _open_properties_dialog(self) -> None:
        if self.properties_window is not None and self.properties_window.winfo_exists():
            self.properties_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Properties")
        win.geometry("420x820")
        win.resizable(False, False)
        win.transient(self.root)
        self.properties_window = win

        auto_edit_var = tk.BooleanVar(value=self.auto_edit_var.get())
        remove_fillers_var = tk.BooleanVar(value=self.remove_fillers_var.get())
        remove_habits_var = tk.BooleanVar(value=self.remove_habits_var.get())
        business_email_var = tk.BooleanVar(value=self.business_email_var.get())
        system_wide_var = tk.BooleanVar(value=self.system_wide_input_var.get())
        llm_enabled_var = tk.BooleanVar(value=self.llm_enabled_var.get())
        whisper_model_id_var = tk.StringVar(value=self.whisper_model_id_var.get())
        whisper_device_var = tk.StringVar(value=self.whisper_device_var.get())
        audio_device_var = tk.StringVar(value=self.audio_device_var.get())
        voice_threshold_var = tk.StringVar(value=self.voice_threshold_var.get())
        brightness_var = tk.StringVar(value="")
        night_light_strength_var = tk.StringVar(value="")
        display_status_var = tk.StringVar(value="Display controls: loading...")

        frame = tk.Frame(win, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Checkbutton(frame, text="Auto edit", variable=auto_edit_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Remove fillers", variable=remove_fillers_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Remove habits", variable=remove_habits_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Convert to business email", variable=business_email_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(frame, text="Enable LLM correction", variable=llm_enabled_var).pack(anchor=tk.W, pady=4)
        tk.Checkbutton(
            frame,
            text="System-wide input (paste to active app on completion)",
            variable=system_wide_var,
        ).pack(anchor=tk.W, pady=4)

        tk.Label(frame, text="Audio input device").pack(anchor=tk.W, pady=(8, 0))
        ttk.Combobox(
            frame, textvariable=audio_device_var,
            values=get_input_device_choices(), state="readonly",
        ).pack(anchor=tk.W, fill=tk.X)

        tk.Label(
            frame, text="Voice threshold (Continuous mode: increase if silence not detected, e.g. 0.01-0.1)"
        ).pack(anchor=tk.W, pady=(8, 0))
        tk.Entry(frame, textvariable=voice_threshold_var, width=10).pack(anchor=tk.W)
        tk.Button(
            frame, text="Auto Adjust Voice Threshold",
            command=lambda: self._auto_adjust_voice_threshold_clicked(
                voice_threshold_var=voice_threshold_var,
                audio_device_value=audio_device_var.get(),
            ),
            bg="#6f42c1", fg="#ffffff", activebackground="#8250df", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=4, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(anchor=tk.W, pady=(6, 0))

        tk.Label(frame, text="ASR model").pack(anchor=tk.W, pady=(8, 0))
        ttk.Combobox(
            frame, textvariable=whisper_model_id_var,
            values=list(get_supported_model_ids()), state="readonly",
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(frame, text="ASR device").pack(anchor=tk.W, pady=(8, 0))
        tk.OptionMenu(frame, whisper_device_var, "gpu", "cpu", "npu").pack(anchor=tk.W, fill=tk.X)

        self.asr_download_button = tk.Button(
            frame, text="Download ASR Model",
            command=lambda: self._download_asr_model_clicked(
                model_id=whisper_model_id_var.get().strip() or "Qwen/Qwen3-ASR-1.7B",
                device=whisper_device_var.get().strip() or "gpu",
            ),
            bg="#1f6feb", fg="#ffffff", activebackground="#2f81f7", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=4, font=("Consolas", 9, "bold"), cursor="hand2",
        )
        self.asr_download_button.pack(anchor=tk.W, pady=(8, 0))
        self.asr_download_progressbar = ttk.Progressbar(
            frame, mode="indeterminate", orient=tk.HORIZONTAL, length=360,
        )
        self.asr_download_progressbar.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))
        tk.Label(
            frame, textvariable=self.asr_download_progress_var,
            anchor="w", justify=tk.LEFT, font=("Consolas", 8), fg="#5a7a9b",
        ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
        if self._asr_download_in_progress:
            self.asr_download_button.config(state=tk.DISABLED)
            self.asr_download_progressbar.start(12)

        tk.Button(
            frame, text="Download LLM Model", command=self._download_model_clicked,
            bg="#2ea043", fg="#ffffff", activebackground="#3fb950", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=4, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(anchor=tk.W, pady=(10, 0))

        self._build_display_section(frame, brightness_var, night_light_strength_var, display_status_var)

        dict_frame = tk.Frame(frame, highlightthickness=1, highlightbackground="#273142")
        dict_frame.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            dict_frame, text="Personal Dictionary (reading -> surface)",
            anchor="w", font=("Consolas", 9, "bold"),
        ).pack(fill=tk.X, padx=6, pady=(6, 2))

        dict_form = tk.Frame(dict_frame)
        dict_form.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(dict_form, text="Reading", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_reading_entry = tk.Entry(dict_form, width=10, relief=tk.FLAT)
        self.dict_reading_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(dict_form, text="Surface", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_surface_entry = tk.Entry(dict_form, width=10, relief=tk.FLAT)
        self.dict_surface_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Button(dict_form, text="Add", command=self._add_dictionary_entry,
                  bg="#2ea043", fg="#ffffff", relief=tk.FLAT, padx=8).pack(side=tk.LEFT)
        tk.Button(dict_form, text="Remove", command=self._remove_dictionary_entry,
                  bg="#b62324", fg="#ffffff", relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=(6, 0))

        self.dict_list = tk.Listbox(dict_frame, height=4, relief=tk.FLAT, font=("Consolas", 9))
        self.dict_list.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.dict_list.bind("<<ListboxSelect>>", self._on_dictionary_selected)
        self._refresh_dictionary_list()

        def apply_and_close() -> None:
            self.auto_edit_var.set(auto_edit_var.get())
            self.remove_fillers_var.set(remove_fillers_var.get())
            self.remove_habits_var.set(remove_habits_var.get())
            self.business_email_var.set(business_email_var.get())
            self.llm_enabled_var.set(llm_enabled_var.get())
            self.llm_defaults["enabled"] = bool(llm_enabled_var.get())
            self.whisper_model_id_var.set(whisper_model_id_var.get())
            self.whisper_device_var.set(whisper_device_var.get())
            self._apply_asr_settings()

            chosen_device = audio_device_var.get()
            self.audio_device_var.set(chosen_device)
            self.asr_defaults["audio_input_device"] = chosen_device
            self.recorder.config.device = parse_device_choice(chosen_device)

            try:
                new_threshold = float(voice_threshold_var.get())
                if new_threshold > 0:
                    self.voice_threshold_var.set(str(new_threshold))
                    self.asr_defaults["voice_threshold"] = new_threshold
                    self.continuous_listener.voice_threshold = new_threshold
            except ValueError:
                pass

            before = self.system_wide_input_var.get()
            after = system_wide_var.get()
            self.system_wide_input_var.set(after)
            if before != after:
                self._toggle_system_wide_input()
            self.status_var.set("Properties updated")
            self._clear_dialog_refs()
            win.destroy()

        buttons = tk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(12, 8), before=dict_frame)
        tk.Button(buttons, text="Apply", command=apply_and_close, width=10).pack(side=tk.LEFT)
        tk.Button(buttons, text="Cancel", command=lambda: (self._clear_dialog_refs(), win.destroy()),
                  width=10).pack(side=tk.RIGHT)

        self._refresh_display_controls(brightness_var, night_light_strength_var, display_status_var)
        win.protocol("WM_DELETE_WINDOW", lambda: (self._clear_dialog_refs(), win.destroy()))

    def _clear_dialog_refs(self) -> None:
        self.dict_reading_entry = None
        self.dict_surface_entry = None
        self.dict_list = None
        self.asr_download_button = None
        self.asr_download_progressbar = None
        self.properties_window = None

    def _build_display_section(
        self,
        frame: tk.Frame,
        brightness_var: tk.StringVar,
        night_light_strength_var: tk.StringVar,
        display_status_var: tk.StringVar,
    ) -> None:
        display_frame = tk.Frame(frame, highlightthickness=1, highlightbackground="#273142")
        display_frame.pack(fill=tk.X, pady=(12, 0))
        tk.Label(display_frame, text="Display", anchor="w", font=("Consolas", 9, "bold")).pack(
            fill=tk.X, padx=6, pady=(6, 2)
        )
        tk.Label(display_frame, textvariable=display_status_var, anchor="w", justify=tk.LEFT,
                 font=("Consolas", 8)).pack(fill=tk.X, padx=6, pady=(0, 6))

        brightness_row = tk.Frame(display_frame)
        brightness_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Label(brightness_row, text="Brightness", font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Button(brightness_row, text="-10", width=5,
                  command=lambda: self._change_brightness_clicked(-10, display_status_var)).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(brightness_row, text="+10", width=5,
                  command=lambda: self._change_brightness_clicked(10, display_status_var)).pack(side=tk.LEFT, padx=(0, 8))
        tk.Entry(brightness_row, textvariable=brightness_var, width=6, relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(brightness_row, text="Set", width=5,
                  command=lambda: self._set_brightness_clicked(brightness_var, display_status_var)).pack(side=tk.LEFT, padx=(6, 0))

        night_light_row = tk.Frame(display_frame)
        night_light_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Label(night_light_row, text="Night Light", font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Button(night_light_row, text="On", width=5,
                  command=lambda: self._set_night_light_enabled_clicked(True, display_status_var)).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(night_light_row, text="Off", width=5,
                  command=lambda: self._set_night_light_enabled_clicked(False, display_status_var)).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(night_light_row, text="Toggle", width=6,
                  command=lambda: self._toggle_night_light_clicked(display_status_var)).pack(side=tk.LEFT)

        strength_row = tk.Frame(display_frame)
        strength_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Label(strength_row, text="Warmth %", font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Entry(strength_row, textvariable=night_light_strength_var, width=6, relief=tk.FLAT).pack(side=tk.LEFT, padx=(8, 6))
        tk.Button(strength_row, text="Set", width=5,
                  command=lambda: self._set_night_light_strength_clicked(night_light_strength_var, display_status_var)).pack(side=tk.LEFT)

    def _refresh_display_controls(
        self,
        brightness_var: tk.StringVar,
        night_light_strength_var: tk.StringVar,
        display_status_var: tk.StringVar,
    ) -> None:
        def worker() -> None:
            try:
                state = self.display_controller.get_state()
                brightness_text = "-" if state.brightness is None else f"{state.brightness}"
                enabled_text = {True: "On", False: "Off"}.get(state.night_light_enabled, "-")
                strength_text = "" if state.night_light_strength is None else f"{state.night_light_strength}"
                summary = f"Brightness: {brightness_text}% | Night Light: {enabled_text}"

                def update_ui() -> None:
                    if state.brightness is not None:
                        brightness_var.set(str(state.brightness))
                    if state.night_light_strength is not None:
                        night_light_strength_var.set(strength_text)
                    display_status_var.set(summary)

                self.root.after(0, update_ui)
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, display_status_var.set, f"Display controls unavailable: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _change_brightness_clicked(self, delta: int, display_status_var: tk.StringVar) -> None:
        display_status_var.set("Updating brightness...")
        threading.Thread(
            target=self._change_brightness_dialog_worker, args=(delta, display_status_var), daemon=True,
        ).start()

    def _change_brightness_dialog_worker(self, delta: int, display_status_var: tk.StringVar) -> None:
        try:
            level = self.display_controller.change_brightness(delta)
            self.root.after(0, display_status_var.set, f"Brightness: {level}% | Night Light: unchanged")
            self.root.after(0, self.status_var.set, f"明るさを {level}% にしました")
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, display_status_var.set, f"Brightness update failed: {exc}")

    def _set_brightness_clicked(self, brightness_var: tk.StringVar, display_status_var: tk.StringVar) -> None:
        try:
            level = int(brightness_var.get().strip())
        except ValueError:
            display_status_var.set("Brightness must be a number between 0 and 100")
            return
        display_status_var.set("Updating brightness...")
        threading.Thread(
            target=self._set_brightness_dialog_worker, args=(level, display_status_var), daemon=True,
        ).start()

    def _set_brightness_dialog_worker(self, level: int, display_status_var: tk.StringVar) -> None:
        try:
            actual = self.display_controller.set_brightness(level)
            self.root.after(0, display_status_var.set, f"Brightness: {actual}% | Night Light: unchanged")
            self.root.after(0, self.status_var.set, f"明るさを {actual}% にしました")
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, display_status_var.set, f"Brightness update failed: {exc}")

    def _set_night_light_enabled_clicked(self, enabled: bool, display_status_var: tk.StringVar) -> None:
        display_status_var.set("Updating Night Light...")
        threading.Thread(
            target=self._set_night_light_enabled_dialog_worker, args=(enabled, display_status_var), daemon=True,
        ).start()

    def _set_night_light_enabled_dialog_worker(self, enabled: bool, display_status_var: tk.StringVar) -> None:
        try:
            if enabled:
                self.display_controller.enable_night_light()
            else:
                self.display_controller.disable_night_light()
            label = "On" if enabled else "Off"
            self.root.after(0, display_status_var.set, f"Brightness: unchanged | Night Light: {label}")
            self.root.after(0, self.status_var.set, f"夜間モードを {'オン' if enabled else 'オフ'} にしました")
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, display_status_var.set, f"Night Light update failed: {exc}")

    def _toggle_night_light_clicked(self, display_status_var: tk.StringVar) -> None:
        display_status_var.set("Updating Night Light...")
        threading.Thread(
            target=self._toggle_night_light_dialog_worker, args=(display_status_var,), daemon=True,
        ).start()

    def _toggle_night_light_dialog_worker(self, display_status_var: tk.StringVar) -> None:
        try:
            enabled = self.display_controller.toggle_night_light()
            label = "On" if enabled else "Off"
            self.root.after(0, display_status_var.set, f"Brightness: unchanged | Night Light: {label}")
            self.root.after(0, self.status_var.set, f"夜間モードを {'オン' if enabled else 'オフ'} にしました")
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, display_status_var.set, f"Night Light update failed: {exc}")

    def _set_night_light_strength_clicked(
        self, strength_var: tk.StringVar, display_status_var: tk.StringVar,
    ) -> None:
        try:
            strength = int(strength_var.get().strip())
        except ValueError:
            display_status_var.set("Night Light strength must be a number between 0 and 100")
            return
        display_status_var.set("Updating Night Light...")
        threading.Thread(
            target=self._set_night_light_strength_dialog_worker, args=(strength, display_status_var), daemon=True,
        ).start()

    def _set_night_light_strength_dialog_worker(
        self, strength: int, display_status_var: tk.StringVar,
    ) -> None:
        try:
            actual = self.display_controller.set_night_light_strength(strength)
            self.root.after(0, display_status_var.set, f"Brightness: unchanged | Night Light warmth: {actual}%")
            self.root.after(0, self.status_var.set, f"夜間モードの強さを {actual}% にしました")
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, display_status_var.set, f"Night Light update failed: {exc}")

    def _refresh_dictionary_list(self) -> None:
        if self.dict_list is None or not self.dict_list.winfo_exists():
            return
        self.dict_entries = self.personal_dictionary.list_entries()
        self.dict_list.delete(0, tk.END)
        for item in self.dict_entries:
            self.dict_list.insert(tk.END, f"{item.reading} -> {item.surface} ({item.count})")

    def _on_dictionary_selected(self, event) -> None:  # noqa: ANN001
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
