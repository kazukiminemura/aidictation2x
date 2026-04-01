"""Main window UI construction and initial state loading.

Standalone functions that mutate the VoiceInputApp instance.
Separated from VoiceInputApp to satisfy SRP (build ≠ orchestrate).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from .asr import get_supported_model_ids

if TYPE_CHECKING:
    from .ui_app import VoiceInputApp


def build_main_ui(app: "VoiceInputApp") -> None:
    """Build all widgets on app.root and assign them to app attributes."""
    app.root.title("ASR2X")
    app.root.geometry("430x840")
    app.root.configure(bg="#0a0e14")

    container = tk.Frame(app.root, bg="#0a0e14")
    container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    top_bar = tk.Frame(container, bg="#141b26", highlightthickness=1, highlightbackground="#273142")
    top_bar.pack(fill=tk.X)

    tk.Label(top_bar, text="●", fg="#ff9f1c", bg="#141b26", font=("Consolas", 11, "bold")).pack(
        side=tk.LEFT, padx=(10, 6), pady=8
    )
    tk.Label(top_bar, text="Voice Input", fg="#e6edf3", bg="#141b26",
             font=("Consolas", 11, "bold")).pack(side=tk.LEFT, pady=8)
    tk.Label(top_bar, textvariable=app.status_var, fg="#9fb1c7", bg="#141b26",
             font=("Consolas", 10)).pack(side=tk.LEFT, padx=12, pady=8)

    controls = tk.Frame(container, bg="#0a0e14")
    controls.pack(fill=tk.X, pady=(10, 8))

    app.record_button = tk.Button(
        controls, text="Start Recording", command=app.toggle_recording,
        bg="#1f6feb", fg="#ffffff", activebackground="#2f81f7", activeforeground="#ffffff",
        relief=tk.FLAT, padx=10, pady=6, font=("Consolas", 10, "bold"), cursor="hand2",
    )
    app.record_button.pack(side=tk.LEFT)

    app.continuous_button = tk.Button(
        controls, text="Continuous", command=app._toggle_continuous,
        bg="#2ea043", fg="#ffffff", activebackground="#3fb950", activeforeground="#ffffff",
        relief=tk.FLAT, padx=10, pady=6, font=("Consolas", 10, "bold"), cursor="hand2",
    )
    app.continuous_button.pack(side=tk.LEFT, padx=(8, 0))

    tk.Label(controls, text="Right-click to open Properties", bg="#0a0e14",
             fg="#8b9fb6", font=("Consolas", 9)).pack(side=tk.LEFT, padx=(12, 4))

    system_frame = tk.Frame(container, bg="#0a0e14")
    system_frame.pack(fill=tk.X, pady=(0, 8))
    tk.Label(
        system_frame,
        text="Hotkeys: Ctrl+Space / Ctrl+Shift+Space = record, Ctrl+Alt+Space = continuous",
        fg="#8b9fb6", bg="#0a0e14", anchor="w", font=("Consolas", 9),
    ).pack(fill=tk.X)
    tk.Label(
        system_frame,
        text=(
            "Voice cmds: クリア / コピー / プロパティ / 明るさ 70 / 明るさを上げて / "
            "夜間モード オン / ナイトライト 45 / 検索 [語句] / [語句] を開く / [語句] に飛ぶ / "
            "リンク [URL] / 一つ前へ / 一つ先に / 辞書登録 [読み] [表記] / 辞書削除 [読み]"
        ),
        fg="#5a7a9b", bg="#0a0e14", anchor="w", font=("Consolas", 8),
    ).pack(fill=tk.X)

    tk.Label(container, text="Output", fg="#8b9fb6", bg="#0a0e14",
             anchor="w", font=("Consolas", 9, "bold")).pack(fill=tk.X)

    style = ttk.Style(app.root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Output.TNotebook", background="#0a0e14", borderwidth=0)
    style.configure(
        "Output.TNotebook.Tab", padding=(10, 4), font=("Consolas", 9, "bold"),
        foreground="#dbe6f3", background="#1a2433",
    )
    style.map(
        "Output.TNotebook.Tab",
        foreground=[("selected", "#ffffff")],
        background=[("selected", "#2f81f7")],
    )

    tabs = ttk.Notebook(container, style="Output.TNotebook")
    tabs.pack(fill=tk.BOTH, expand=True)

    asr_tab = tk.Frame(tabs, bg="#0a0e14")
    final_tab = tk.Frame(tabs, bg="#0a0e14")
    tabs.add(asr_tab, text="ASR Text")
    tabs.add(final_tab, text="Final")
    tab_selected_colors = {0: "#14532d", 1: "#1d4ed8"}

    def apply_selected_tab_color() -> None:
        try:
            current_idx = tabs.index("current")
        except tk.TclError:
            current_idx = 0
        selected_bg = tab_selected_colors.get(current_idx, "#2f81f7")
        style.map(
            "Output.TNotebook.Tab",
            foreground=[("selected", "#ffffff")],
            background=[("selected", selected_bg)],
        )

    tabs.bind("<<NotebookTabChanged>>", lambda _event: apply_selected_tab_color())
    apply_selected_tab_color()

    text_kwargs = dict(
        height=18, wrap=tk.WORD, bg="#0b111a", fg="#dbe6f3",
        insertbackground="#dbe6f3", relief=tk.FLAT, font=("Consolas", 9),
    )
    app.asr_text = tk.Text(asr_tab, **text_kwargs)
    app.asr_text.pack(fill=tk.BOTH, expand=True)

    app.final_text = tk.Text(final_tab, **text_kwargs)
    app.final_text.pack(fill=tk.BOTH, expand=True)


def load_initial_state(app: "VoiceInputApp") -> None:
    """Populate text areas from autosave or display the welcome message."""
    auto = app.storage.load_autosave()
    if auto:
        app.current_raw_text = auto.raw_text
        app._set_text(app.final_text, auto.final_text)
        if app.asr_text is not None:
            app._set_text(app.asr_text, auto.raw_text)
        app.status_var.set("Ready (Ctrl+Space / Ctrl+Shift+Space / Ctrl+Alt+Space)")
        return

    if app.asr_text is not None:
        app._set_text(
            app.asr_text,
            (
                "ASR2X is ready.\n\n"
                "Quick start:\n"
                "1. Right-click -> Properties\n"
                "2. Download ASR Model\n"
                "3. Press Start Recording or use Ctrl+Space\n"
                "4. Toggle Continuous with Ctrl+Alt+Space\n\n"
                "Default model: Qwen/Qwen3-ASR-1.7B\n"
                "Other models: Qwen/Qwen3-ASR-0.6B / openai/whisper-base / openai/whisper-large-v3-turbo"
            ),
        )
    app._set_text(
        app.final_text,
        (
            "Your final text will appear here.\n\n"
            "Recommended first run:\n"
            "- Choose Qwen or Whisper IR model in Properties\n"
            "- Turn on LLM correction later if needed"
        ),
    )
    if app.asr_engine.get_model_dir().exists():
        app.status_var.set("Ready (Ctrl+Space / Ctrl+Shift+Space / Ctrl+Alt+Space)")
    else:
        app.status_var.set("Ready - Right-click -> Properties -> Download ASR Model")
