import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from .asr import ASREngine
from .audio_capture import AudioConfig, AudioRecorder
from .personal_dictionary import PersonalDictionary
from .storage import Storage
from .system_wide_input import SystemWideInput
from .text_processing import ProcessOptions, process_text


class VoiceInputApp:
    def __init__(
        self,
        root: tk.Tk,
        asr_engine: ASREngine,
        recorder: AudioRecorder,
        storage: Storage,
        rules: dict,
        personal_dictionary: PersonalDictionary,
        enable_system_wide_input_default: bool,
    ):
        self.root = root
        self.asr_engine = asr_engine
        self.recorder = recorder
        self.storage = storage
        self.rules = rules
        self.personal_dictionary = personal_dictionary
        self.logger = logging.getLogger(__name__)

        self.auto_edit_var = tk.BooleanVar(value=True)
        self.remove_fillers_var = tk.BooleanVar(value=True)
        self.remove_habits_var = tk.BooleanVar(value=True)
        self.system_wide_input_var = tk.BooleanVar(value=enable_system_wide_input_default)
        self.status_var = tk.StringVar(value="起動中...")
        self.current_raw_text = ""
        self.hotkey_pressed = False

        self.system_wide_input = SystemWideInput(
            dispatch_on_ui=lambda cb: self.root.after(0, cb),
            on_toggle=self.toggle_recording,
        )

        self._build_ui()
        self._bind_hotkeys()
        self._load_initial_state()
        self._refresh_dictionary_list()

        if self.system_wide_input_var.get():
            self.system_wide_input.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.title("音声入力アプリ")
        self.root.geometry("430x840")
        self.root.configure(bg="#0a0e14")

        container = tk.Frame(self.root, bg="#0a0e14")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        top_bar = tk.Frame(container, bg="#141b26", highlightthickness=1, highlightbackground="#273142")
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text="●", fg="#ff9f1c", bg="#141b26", font=("Consolas", 11, "bold")).pack(
            side=tk.LEFT, padx=(10, 6), pady=8
        )
        tk.Label(
            top_bar,
            text="音声入力",
            fg="#e6edf3",
            bg="#141b26",
            font=("Consolas", 11, "bold"),
        ).pack(side=tk.LEFT, pady=8)
        tk.Label(
            top_bar,
            textvariable=self.status_var,
            fg="#9fb1c7",
            bg="#141b26",
            font=("Consolas", 10),
        ).pack(side=tk.LEFT, padx=12, pady=8)

        controls = tk.Frame(container, bg="#0a0e14")
        controls.pack(fill=tk.X, pady=(10, 8))

        self.record_button = tk.Button(
            controls,
            text="録音開始",
            command=self.toggle_recording,
            bg="#1f6feb",
            fg="#ffffff",
            activebackground="#2f81f7",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Consolas", 10, "bold"),
            cursor="hand2",
        )
        self.record_button.pack(side=tk.LEFT)

        tk.Checkbutton(
            controls,
            text="自動編集",
            variable=self.auto_edit_var,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=(12, 4))
        tk.Checkbutton(
            controls,
            text="フィラーワード除去",
            variable=self.remove_fillers_var,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(
            controls,
            text="話し癖除去",
            variable=self.remove_habits_var,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=4)

        system_frame = tk.Frame(container, bg="#0a0e14")
        system_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Checkbutton(
            system_frame,
            text="システム横断入力（完了後にアクティブアプリへ貼り付け）",
            variable=self.system_wide_input_var,
            command=self._toggle_system_wide_input,
            bg="#0a0e14",
            fg="#c9d1d9",
            activebackground="#0a0e14",
            activeforeground="#c9d1d9",
            selectcolor="#141b26",
            font=("Consolas", 9),
        ).pack(anchor=tk.W)
        tk.Label(
            system_frame,
            text="グローバルホットキー: Ctrl+Shift+Space",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9),
        ).pack(fill=tk.X)

        dict_frame = tk.Frame(container, bg="#0a0e14", highlightthickness=1, highlightbackground="#273142")
        dict_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            dict_frame,
            text="パーソナル辞書（読み -> 表記）",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9, "bold"),
        ).pack(fill=tk.X, padx=6, pady=(6, 2))

        form = tk.Frame(dict_frame, bg="#0a0e14")
        form.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(form, text="読み", fg="#c9d1d9", bg="#0a0e14", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_reading_entry = tk.Entry(
            form,
            width=10,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
        )
        self.dict_reading_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(form, text="表記", fg="#c9d1d9", bg="#0a0e14", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.dict_surface_entry = tk.Entry(
            form,
            width=10,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
        )
        self.dict_surface_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Button(
            form,
            text="学習登録",
            command=self._add_dictionary_entry,
            bg="#2ea043",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
        ).pack(side=tk.LEFT)
        tk.Button(
            form,
            text="選択削除",
            command=self._remove_dictionary_entry,
            bg="#b62324",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.dict_list = tk.Listbox(
            dict_frame,
            height=4,
            bg="#0b111a",
            fg="#dbe6f3",
            selectbackground="#1f6feb",
            selectforeground="#ffffff",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.dict_list.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.dict_list.bind("<<ListboxSelect>>", self._on_dictionary_selected)

        final_title = tk.Label(
            container,
            text="最終結果",
            fg="#8b9fb6",
            bg="#0a0e14",
            anchor="w",
            font=("Consolas", 9, "bold"),
        )
        final_title.pack(fill=tk.X)

        self.final_text = tk.Text(
            container,
            height=18,
            wrap=tk.WORD,
            bg="#0b111a",
            fg="#dbe6f3",
            insertbackground="#dbe6f3",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.final_text.pack(fill=tk.BOTH, expand=True)

    def _load_initial_state(self) -> None:
        auto = self.storage.load_autosave()
        if auto:
            self.current_raw_text = auto.raw_text
            self.final_text.insert("1.0", auto.final_text)
        self.status_var.set("待機中（Ctrl+Space / Ctrl+Shift+Space）")

    def _bind_hotkeys(self) -> None:
        self.root.bind_all("<Control-KeyPress-space>", self._on_hotkey_press)
        self.root.bind_all("<Control-KeyRelease-space>", self._on_hotkey_release)

    def _on_hotkey_press(self, event):  # noqa: ANN001
        if event.state & 0x0001:
            return
        if self.hotkey_pressed:
            return "break"
        self.hotkey_pressed = True
        self.toggle_recording()
        return "break"

    def _on_hotkey_release(self, event):  # noqa: ANN001, ARG002
        self.hotkey_pressed = False
        return "break"

    def _toggle_system_wide_input(self) -> None:
        if self.system_wide_input_var.get():
            self.system_wide_input.start()
            self.status_var.set("システム横断入力: ON")
        else:
            self.system_wide_input.stop()
            self.status_var.set("システム横断入力: OFF")

    def _refresh_dictionary_list(self) -> None:
        self.dict_entries = self.personal_dictionary.list_entries()
        self.dict_list.delete(0, tk.END)
        for item in self.dict_entries:
            self.dict_list.insert(tk.END, f"{item.reading} -> {item.surface} ({item.count})")

    def _on_dictionary_selected(self, event):  # noqa: ANN001
        if not self.dict_list.curselection():
            return
        idx = self.dict_list.curselection()[0]
        item = self.dict_entries[idx]
        self.dict_reading_entry.delete(0, tk.END)
        self.dict_reading_entry.insert(0, item.reading)
        self.dict_surface_entry.delete(0, tk.END)
        self.dict_surface_entry.insert(0, item.surface)

    def _add_dictionary_entry(self) -> None:
        try:
            self.personal_dictionary.add_or_update(
                reading=self.dict_reading_entry.get(),
                surface=self.dict_surface_entry.get(),
            )
        except ValueError as exc:
            messagebox.showwarning("入力不足", str(exc))
            return
        self._refresh_dictionary_list()
        self.status_var.set("辞書を学習登録しました")

    def _remove_dictionary_entry(self) -> None:
        reading = self.dict_reading_entry.get().strip()
        if not reading:
            messagebox.showwarning("削除対象なし", "削除する読みを選択してください。")
            return
        self.personal_dictionary.remove(reading)
        self._refresh_dictionary_list()
        self.status_var.set("辞書から削除しました")

    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            try:
                self.recorder.start()
                self.record_button.config(text="録音停止", bg="#b62324", activebackground="#d73a49")
                self.status_var.set("録音中")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("録音エラー", str(exc))
                self.logger.exception("Failed to start recording")
            return

        self.record_button.config(text="録音開始", bg="#1f6feb", activebackground="#2f81f7")
        self.status_var.set("文字起こし中")

        audio = self.recorder.stop()
        threading.Thread(target=self._transcribe_and_process, args=(audio,), daemon=True).start()

    def _transcribe_and_process(self, audio_data) -> None:  # noqa: ANN001
        try:
            raw = self.asr_engine.transcribe(audio_data)
            raw = self.personal_dictionary.apply(raw)
            options = ProcessOptions(
                auto_edit=self.auto_edit_var.get(),
                remove_fillers=self.remove_fillers_var.get(),
                remove_habits=self.remove_habits_var.get(),
            )
            final = process_text(raw, self.rules, options)
            self.storage.save_autosave(raw, final)
            self.storage.append_history(raw, final)
            self.root.after(0, self._apply_results, raw, final, "")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            self.root.after(0, self._apply_results, "", "", str(exc))

    def _apply_results(self, raw: str, final: str, error: str) -> None:
        if error:
            self.status_var.set("エラー")
            messagebox.showerror("処理エラー", error)
            return

        self._set_text(self.final_text, final)
        self.current_raw_text = raw
        if self.system_wide_input_var.get():
            try:
                self.system_wide_input.paste_to_active_app(final)
                self.status_var.set("完了（他アプリへ入力）")
            except Exception as exc:  # noqa: BLE001
                self.status_var.set("完了（貼り付け失敗）")
                messagebox.showwarning("貼り付け失敗", str(exc))
        else:
            self.status_var.set("完了")

    def _on_close(self) -> None:
        self.system_wide_input.stop()
        self.root.destroy()

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)


def build_app(
    root: tk.Tk,
    model_dir: Path,
    audio_config: AudioConfig,
    storage: Storage,
    rules: dict,
    personal_dictionary: PersonalDictionary,
    enable_system_wide_input_default: bool,
) -> VoiceInputApp:
    engine = ASREngine(model_dir=model_dir, sample_rate_hz=audio_config.sample_rate_hz)
    recorder = AudioRecorder(config=audio_config)
    return VoiceInputApp(
        root=root,
        asr_engine=engine,
        recorder=recorder,
        storage=storage,
        rules=rules,
        personal_dictionary=personal_dictionary,
        enable_system_wide_input_default=enable_system_wide_input_default,
    )
