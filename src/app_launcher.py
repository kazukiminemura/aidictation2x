"""Windows application discovery and launching via registry and Start Menu."""
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Japanese/alias → executable stem used in App Paths registry
_ALIASES: dict[str, str] = {
    "パワーポイント": "powerpnt",
    "パワポ": "powerpnt",
    "エクセル": "excel",
    "ワード": "winword",
    "メモ帳": "notepad",
    "エクスプローラー": "explorer",
    "タスクマネージャー": "taskmgr",
    "タスクマネージャ": "taskmgr",
    "コントロールパネル": "control",
    "電卓": "calc",
    "ペイント": "mspaint",
}

_BROWSER_NAMES = {
    "ブラウザ", "browser",
    "chrome", "クローム",
    "edge", "エッジ",
    "firefox", "ファイヤーフォックス",
}

# Start Menu entries whose names suggest they are not launchers
_SKIP_KEYWORDS = (
    "uninstall", "アンインストール", "readme", "help",
    "manual", "マニュアル", "release notes", "リリースノート",
)


@dataclass
class AppEntry:
    name: str
    path: str  # absolute path to .exe or .lnk


class AppLauncher:
    """Discovers installed Windows apps and launches them by name query."""

    def __init__(self) -> None:
        self._cache: list[AppEntry] | None = None

    def refresh(self) -> None:
        """Invalidate the app cache so it is rebuilt on next access."""
        self._cache = None

    def launch(self, query: str) -> str:
        """Find an app matching *query* and start it. Returns display name."""
        import webbrowser

        q = query.strip()
        q_lower = q.lower()

        if q_lower in _BROWSER_NAMES:
            webbrowser.open("about:blank")
            return "ブラウザ"

        # Resolve Japanese/common aliases to registry exe stem
        resolved = _ALIASES.get(q_lower, q)

        entry = self._find(resolved)
        if entry is None and resolved != q:
            # Alias resolved but not found; try original text
            entry = self._find(q)
        if entry is None:
            raise ValueError(f"アプリが見つかりません: {query}")

        os.startfile(entry.path)
        return entry.name

    def close(self, query: str) -> str:
        """Terminate running processes matching *query*. Returns display name."""
        import subprocess

        q = query.strip()
        q_lower = q.lower()

        if q_lower in _BROWSER_NAMES:
            for proc in ("chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "brave.exe"):
                subprocess.run(["taskkill", "/IM", proc, "/F"], capture_output=True)
            return "ブラウザ"

        resolved = _ALIASES.get(q_lower, q)
        entry = self._find(resolved)
        if entry is None and resolved != q:
            entry = self._find(q)
        if entry is None:
            raise ValueError(f"アプリが見つかりません: {query}")

        exe_name = self._get_exe_name(entry, resolved)
        result = subprocess.run(
            ["taskkill", "/IM", exe_name, "/F"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"起動していません: {entry.name}")
        return entry.name

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_exe_name(self, entry: AppEntry, fallback: str) -> str:
        path = Path(entry.path)
        if path.suffix.lower() == ".exe":
            return path.name
        # .lnk — use the app name or fallback stem as best-effort process name
        stem = fallback or entry.name
        return f"{stem}.exe"

    def _find(self, query: str) -> Optional[AppEntry]:
        import difflib

        q = query.strip().lower()
        if not q:
            return None
        entries = self._get_entries()
        names_lower = [e.name.lower() for e in entries]

        # 1. Exact match
        for e in entries:
            if e.name.lower() == q:
                return e

        # 2. Starts-with (prefer shortest)
        sw = [e for e in entries if e.name.lower().startswith(q)]
        if sw:
            return min(sw, key=lambda e: len(e.name))

        # 3. Contains (prefer shortest)
        co = [e for e in entries if q in e.name.lower()]
        if co:
            return min(co, key=lambda e: len(e.name))

        # 4. Query is contained in a word token of the entry name
        #    e.g. query="excel" matches "Microsoft Excel 365"
        for e in entries:
            tokens = re.split(r'[\s\-_]+', e.name.lower())
            if any(q == tok or tok.startswith(q) for tok in tokens):
                return e

        # 5. difflib fuzzy match (cutoff 0.6 to avoid wrong guesses)
        close = difflib.get_close_matches(q, names_lower, n=1, cutoff=0.6)
        if close:
            idx = names_lower.index(close[0])
            return entries[idx]

        return None

    def _get_entries(self) -> list[AppEntry]:
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def _build(self) -> list[AppEntry]:
        entries: list[AppEntry] = []
        seen: set[str] = set()

        def _add(e: AppEntry) -> None:
            key = e.name.lower()
            if key not in seen:
                seen.add(key)
                entries.append(e)

        for e in self._scan_start_menu():
            _add(e)
        for e in self._scan_registry():
            _add(e)

        logger.debug("AppLauncher: %d apps discovered", len(entries))
        return entries

    def _scan_start_menu(self) -> list[AppEntry]:
        entries: list[AppEntry] = []
        dirs = [
            Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path("C:/ProgramData/Microsoft/Windows/Start Menu/Programs"),
        ]
        for d in dirs:
            if not d.exists():
                continue
            for lnk in d.rglob("*.lnk"):
                name = lnk.stem
                if any(skip in name.lower() for skip in _SKIP_KEYWORDS):
                    continue
                entries.append(AppEntry(name=name, path=str(lnk)))
        return entries

    def _scan_registry(self) -> list[AppEntry]:
        try:
            import winreg
        except ImportError:
            return []

        entries: list[AppEntry] = []
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(
                    hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
                )
            except OSError:
                continue
            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, i)
                except OSError:
                    break
                i += 1
                try:
                    subkey = winreg.OpenKey(key, subkey_name)
                    try:
                        raw_path, _ = winreg.QueryValueEx(subkey, "")
                        path = str(raw_path).strip().strip('"')
                        if path and Path(path).exists():
                            entries.append(AppEntry(name=Path(subkey_name).stem, path=path))
                    except OSError:
                        pass
                    finally:
                        winreg.CloseKey(subkey)
                except OSError:
                    pass
            winreg.CloseKey(key)
        return entries
