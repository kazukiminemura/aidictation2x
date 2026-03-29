"""Voice command detection for hands-free app operation."""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceCommand:
    action: str
    args: dict = field(default_factory=dict)


def detect_voice_command(text: str) -> Optional[VoiceCommand]:
    """Return a VoiceCommand if *text* matches a known command pattern, else None."""
    t = text.strip()
    if not t:
        return None

    # --- Dictionary add ---
    # "辞書登録 reading surface"  /  "単語登録 reading surface"
    m = re.match(r'^(?:辞書登録|単語登録)\s+(\S+)\s+(.+)$', t)
    if m:
        return VoiceCommand("dict_add", {"reading": m.group(1).strip(), "surface": m.group(2).strip()})

    # "readingをsurfaceに登録"  /  "readingをsurfaceに追加"
    m = re.match(r'^(\S+)\s*を\s*(\S+)\s*に(?:登録|追加)$', t)
    if m:
        return VoiceCommand("dict_add", {"reading": m.group(1).strip(), "surface": m.group(2).strip()})

    # --- Dictionary remove ---
    # "辞書削除 reading"  /  "単語削除 reading"
    m = re.match(r'^(?:辞書削除|単語削除)\s+(\S+)$', t)
    if m:
        return VoiceCommand("dict_remove", {"reading": m.group(1).strip()})

    # "readingを辞書から削除"
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

    # --- Open apps ---
    if re.match(r'^(?:パワーポイント|パワポ|PowerPoint)(?:を開く|開いて|起動)?$', t, re.IGNORECASE):
        return VoiceCommand("open_app", {"app": "powerpoint"})

    if re.match(r'^(?:エクセル|Excel)(?:を開く|開いて|起動)?$', t, re.IGNORECASE):
        return VoiceCommand("open_app", {"app": "excel"})

    if re.match(r'^(?:ワード|Word)(?:を開く|開いて|起動)?$', t, re.IGNORECASE):
        return VoiceCommand("open_app", {"app": "word"})

    if re.match(r'^(?:ブラウザ|ブラウザを開く|Chrome|クローム|Edge|エッジ|Firefox|ファイヤーフォックス)(?:を開く|開いて|起動)?$', t, re.IGNORECASE):
        return VoiceCommand("open_app", {"app": "browser"})

    return None
