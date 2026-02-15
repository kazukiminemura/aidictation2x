import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .asr import ASREngine
from .audio_capture import AudioConfig, AudioRecorder
from .storage import Storage
from .text_processing import ProcessOptions, create_diff_text, process_text


class VoiceInputApp:
    def __init__(
        self,
        root: tk.Tk,
        asr_engine: ASREngine,
        recorder: AudioRecorder,
        storage: Storage,
        rules: dict,
    ):
        self.root = root
        self.asr_engine = asr_engine
        self.recorder = recorder
        self.storage = storage
        self.rules = rules
        self.logger = logging.getLogger(__name__)

        self.auto_edit_var = tk.BooleanVar(value=True)
        self.remove_fillers_var = tk.BooleanVar(value=True)
        self.remove_habits_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="待機中")
        self._build_ui()
        self._load_initial_state()

    def _build_ui(self) -> None:
        self.root.title("音声入力アプリ MVP")
        self.root.geometry("980x700")

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        self.record_button = ttk.Button(top, text="録音開始", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        options = ttk.LabelFrame(self.root, text="処理オプション", padding=8)
        options.pack(fill=tk.X, padx=8, pady=6)
        ttk.Checkbutton(options, text="自動編集", variable=self.auto_edit_var).pack(side=tk.LEFT)
        ttk.Checkbutton(
            options, text="フィラーワード除去", variable=self.remove_fillers_var
        ).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(options, text="話し癖除去", variable=self.remove_habits_var).pack(side=tk.LEFT)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=1)

        ttk.Label(left, text="ASR生テキスト").pack(anchor=tk.W)
        self.raw_text = tk.Text(left, height=16, wrap=tk.WORD)
        self.raw_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="最終テキスト").pack(anchor=tk.W)
        self.final_text = tk.Text(right, height=16, wrap=tk.WORD)
        self.final_text.pack(fill=tk.BOTH, expand=True)

        diff_frame = ttk.LabelFrame(self.root, text="差分", padding=6)
        diff_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.diff_text = tk.Text(diff_frame, height=10, wrap=tk.NONE)
        self.diff_text.pack(fill=tk.BOTH, expand=True)

        history_frame = ttk.LabelFrame(self.root, text="履歴（最新10件）", padding=6)
        history_frame.pack(fill=tk.BOTH, padx=8, pady=6)
        self.history_list = tk.Listbox(history_frame, height=8)
        self.history_list.pack(fill=tk.BOTH, expand=True)
        self.history_list.bind("<<ListboxSelect>>", self.on_history_selected)

    def _load_initial_state(self) -> None:
        auto = self.storage.load_autosave()
        if auto:
            self.raw_text.insert("1.0", auto.raw_text)
            self.final_text.insert("1.0", auto.final_text)
            self._refresh_diff()
        self._refresh_history()

    def _refresh_history(self) -> None:
        self.history_items = self.storage.load_history()
        self.history_list.delete(0, tk.END)
        for item in self.history_items:
            self.history_list.insert(tk.END, item.timestamp)

    def on_history_selected(self, event):  # noqa: ANN001
        if not self.history_list.curselection():
            return
        idx = self.history_list.curselection()[0]
        item = self.history_items[idx]
        self._set_text(self.raw_text, item.raw_text)
        self._set_text(self.final_text, item.final_text)
        self._refresh_diff()

    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            try:
                self.recorder.start()
                self.record_button.config(text="録音停止")
                self.status_var.set("録音中...")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("録音エラー", str(exc))
                self.logger.exception("Failed to start recording")
            return
        self.record_button.config(text="録音開始")
        self.status_var.set("文字起こし中...")

        audio = self.recorder.stop()
        threading.Thread(target=self._transcribe_and_process, args=(audio,), daemon=True).start()

    def _transcribe_and_process(self, audio_data) -> None:  # noqa: ANN001
        try:
            raw = self.asr_engine.transcribe(audio_data)
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
        self._set_text(self.raw_text, raw)
        self._set_text(self.final_text, final)
        self._refresh_diff()
        self._refresh_history()
        self.status_var.set("完了")

    def _refresh_diff(self) -> None:
        raw = self.raw_text.get("1.0", tk.END).strip()
        final = self.final_text.get("1.0", tk.END).strip()
        diff = create_diff_text(raw, final)
        self._set_text(self.diff_text, diff)

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
) -> VoiceInputApp:
    engine = ASREngine(model_dir=model_dir, sample_rate_hz=audio_config.sample_rate_hz)
    recorder = AudioRecorder(config=audio_config)
    return VoiceInputApp(root=root, asr_engine=engine, recorder=recorder, storage=storage, rules=rules)
