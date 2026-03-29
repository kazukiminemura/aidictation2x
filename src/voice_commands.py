"""Voice command detection for hands-free app operation."""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceCommand:
    action: str
    args: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action word groups (used in multiple patterns)
# ---------------------------------------------------------------------------

# 起動系
_LAUNCH = (
    r'(?:起動|開く|開いて|立ち上げ(?:る|て)?'
    r'|スタート(?:する|して)?|ランチ(?:する|して)?'
    r'|動かし(?:て|てください)?|使(?:う|って))'
)

# 終了系
_CLOSE = (
    r'(?:終了|閉じ(?:る|て)|終わ(?:る|って)|止め(?:る|て)'
    r'|停止(?:する|して)?|落とし(?:て|てください)?'
    r'|消し(?:て|てください)?|終わらせ(?:て|てください)?'
    r'|シャットダウン(?:する|して)?)'
)

# 後置助動詞: "する" "して" "してください" (optional)
_DO = r'(?:する|して(?:ください)?)?'


def detect_voice_command(text: str) -> Optional[VoiceCommand]:
    """Return a VoiceCommand if *text* matches a known command pattern, else None."""
    # Strip trailing punctuation and whitespace before matching
    t = re.sub(r'[。、！？…,.!?\s]+$', '', text).strip()
    if not t:
        return None

    # --- Dictionary add ---
    m = re.match(r'^(?:辞書登録|単語登録)\s+(\S+)\s+(.+)$', t)
    if m:
        return VoiceCommand("dict_add", {"reading": m.group(1).strip(), "surface": m.group(2).strip()})

    m = re.match(r'^(\S+)\s*を\s*(\S+)\s*に(?:登録|追加)$', t)
    if m:
        return VoiceCommand("dict_add", {"reading": m.group(1).strip(), "surface": m.group(2).strip()})

    # --- Dictionary remove ---
    m = re.match(r'^(?:辞書削除|単語削除)\s+(\S+)$', t)
    if m:
        return VoiceCommand("dict_remove", {"reading": m.group(1).strip()})

    m = re.match(r'^(\S+)\s*を辞書から(?:削除|消去)$', t)
    if m:
        return VoiceCommand("dict_remove", {"reading": m.group(1).strip()})

    # --- Clear text ---
    if re.match(r'^(?:クリア|テキストクリア|全部消す|全文削除|テキスト削除)$', t):
        return VoiceCommand("clear", {})

    # --- Copy text ---
    if re.match(r'^(?:コピー|テキストコピー)$', t):
        return VoiceCommand("copy", {})

    # --- Open properties ---
    if re.match(r'^(?:プロパティ|設定を開く|設定|プロパティを開く)$', t):
        return VoiceCommand("properties", {})

    # --- Launch any installed app ---
    # Pattern A: action word first — "起動 エクセル" / "開いて メモ帳"
    m = re.match(r'^(?:起動|開いて|立ち上げて?)\s+(.+)$', t)
    if m:
        return VoiceCommand("launch_any", {"query": m.group(1).strip()})

    # Pattern B: app name first, action word last (を optional, spaces optional)
    #   "エクセル起動" / "エクセルを起動" / "エクセル 起動して" / "エクセルを開いて"
    m = re.match(r'^(.+?)\s*を?\s*' + _LAUNCH + _DO + r'$', t)
    if m:
        q = m.group(1).strip()
        if q:
            return VoiceCommand("launch_any", {"query": q})

    # --- Close / terminate any app ---
    # Pattern A: action word first — "終了 エクセル" / "閉じて メモ帳"
    m = re.match(r'^(?:終了|閉じ(?:て|る)|止め(?:て|る))\s+(.+)$', t)
    if m:
        return VoiceCommand("close_app", {"query": m.group(1).strip()})

    # Pattern B: app name first, action word last (を optional)
    #   "エクセル終了" / "エクセルを終了して" / "メモ帳閉じて" / "エクセル 止めて"
    m = re.match(r'^(.+?)\s*を?\s*' + _CLOSE + _DO + r'$', t)
    if m:
        q = m.group(1).strip()
        if q:
            return VoiceCommand("close_app", {"query": q})

    return None
