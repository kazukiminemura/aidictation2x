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


def _looks_like_url_target(text: str) -> bool:
    candidate = text.strip().lower()
    if not candidate:
        return False
    return (
        candidate.startswith(("http://", "https://", "www."))
        or "." in candidate
    )


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

    # --- Display brightness ---
    m = re.match(r'^(?:明るさ|画面の明るさ)\s*(?:を)?\s*(\d{1,3})\s*(?:%|パーセント)?\s*(?:にして|にする|へ変更|へして)?$', t)
    if m:
        return VoiceCommand("display_brightness_set", {"level": int(m.group(1))})

    if re.match(r'^(?:明るさ|画面の明るさ)\s*(?:を)?\s*(?:上げて|明るくして|アップ|増やして)$', t):
        return VoiceCommand("display_brightness_adjust", {"delta": 10})

    if re.match(r'^(?:明るさ|画面の明るさ)\s*(?:を)?\s*(?:下げて|暗くして|ダウン|減らして)$', t):
        return VoiceCommand("display_brightness_adjust", {"delta": -10})

    # --- Night Light ---
    m = re.match(r'^(?:夜間モード|ナイトライト)\s*(?:を)?\s*(\d{1,3})\s*(?:%|パーセント)?\s*(?:にして|にする|へ変更|へして)?$', t)
    if m:
        return VoiceCommand("night_light_strength_set", {"strength": int(m.group(1))})

    if re.match(r'^(?:夜間モード|ナイトライト)\s*(?:を)?\s*(?:オン|つけて|有効|有効にして)$', t):
        return VoiceCommand("night_light_set_enabled", {"enabled": True})

    if re.match(r'^(?:夜間モード|ナイトライト)\s*(?:を)?\s*(?:オフ|消して|無効|無効にして)$', t):
        return VoiceCommand("night_light_set_enabled", {"enabled": False})

    if re.match(r'^(?:夜間モード|ナイトライト)\s*(?:を)?\s*(?:切り替え|トグル)$', t):
        return VoiceCommand("night_light_toggle", {})

    # --- Browser navigation ---
    if re.match(r'^(?:ブラウザ)?\s*(?:戻る|前へ|ひとつ前へ|一つ前へ)$', t):
        return VoiceCommand("browser_back", {})

    if re.match(r'^(?:ブラウザ)?\s*(?:進む|次へ|ひとつ先に|一つ先に|ひとつ先へ|一つ先へ)$', t):
        return VoiceCommand("browser_forward", {})

    # --- Browser search ---
    m = re.match(r'^(?:ブラウザで)?検索\s+(.+)$', t)
    if m:
        return VoiceCommand("browser_search", {"query": m.group(1).strip()})

    m = re.match(r'^(.+?)\s*を\s*(?:ブラウザで)?検索$', t)
    if m:
        query = m.group(1).strip()
        if query:
            return VoiceCommand("browser_search", {"query": query})

    # --- Browser open top search result ---
    m = re.match(r'^(.+?)\s*を\s*(?:ブラウザで)?開(?:く|いて)$', t)
    if m:
        query = m.group(1).strip()
        if query and not _looks_like_url_target(query):
            return VoiceCommand("browser_open_result", {"query": query})

    m = re.match(r'^(.+?)\s*(?:の)?(?:リンク|検索結果)\s*(?:を)?開(?:く|いて)$', t)
    if m:
        query = m.group(1).strip()
        if query:
            return VoiceCommand("browser_open_result", {"query": query})

    # --- Browser open URL ---
    m = re.match(r'^(?:リンク(?:先)?|url)\s+(.+)$', t, flags=re.IGNORECASE)
    if m:
        target = m.group(1).strip()
        if target:
            target = re.sub(r'\s*(?:を)?(?:開く|開いて|に飛ぶ|へ飛ぶ)$', '', target).strip()
            if _looks_like_url_target(target):
                return VoiceCommand("browser_open_url", {"target": target})

    m = re.match(r'^(.+?)\s*(?:に|へ)\s*飛(?:ぶ|んで)$', t)
    if m:
        target = m.group(1).strip()
        if _looks_like_url_target(target):
            return VoiceCommand("browser_open_url", {"target": target})
        if target:
            return VoiceCommand("browser_open_result", {"query": target})

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
