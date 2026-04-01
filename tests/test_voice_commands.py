from src.voice_commands import detect_voice_command


def test_detect_browser_search_prefix() -> None:
    cmd = detect_voice_command("検索 天気予報")

    assert cmd is not None
    assert cmd.action == "browser_search"
    assert cmd.args == {"query": "天気予報"}


def test_detect_browser_search_suffix() -> None:
    cmd = detect_voice_command("渋谷 ランチ を検索")

    assert cmd is not None
    assert cmd.action == "browser_search"
    assert cmd.args == {"query": "渋谷 ランチ"}


def test_detect_browser_back_command() -> None:
    cmd = detect_voice_command("一つ前へ")

    assert cmd is not None
    assert cmd.action == "browser_back"


def test_detect_browser_forward_command() -> None:
    cmd = detect_voice_command("一つ先に")

    assert cmd is not None
    assert cmd.action == "browser_forward"


def test_detect_browser_open_link_command() -> None:
    cmd = detect_voice_command("リンク example.com")

    assert cmd is not None
    assert cmd.action == "browser_open_url"
    assert cmd.args == {"target": "example.com"}


def test_detect_browser_open_fly_command() -> None:
    cmd = detect_voice_command("https://openai.com に飛ぶ")

    assert cmd is not None
    assert cmd.action == "browser_open_url"
    assert cmd.args == {"target": "https://openai.com"}


def test_detect_browser_open_result_command() -> None:
    cmd = detect_voice_command("OpenAI を開く")

    assert cmd is not None
    assert cmd.action == "browser_open_result"
    assert cmd.args == {"query": "OpenAI"}


def test_detect_browser_open_result_fly_command() -> None:
    cmd = detect_voice_command("OpenAI に飛ぶ")

    assert cmd is not None
    assert cmd.action == "browser_open_result"
    assert cmd.args == {"query": "OpenAI"}


def test_detect_display_brightness_adjust_up() -> None:
    cmd = detect_voice_command("明るさを上げて")

    assert cmd is not None
    assert cmd.action == "display_brightness_adjust"
    assert cmd.args == {"delta": 10}


def test_detect_display_brightness_set() -> None:
    cmd = detect_voice_command("明るさ 70")

    assert cmd is not None
    assert cmd.action == "display_brightness_set"
    assert cmd.args == {"level": 70}


def test_detect_night_light_on() -> None:
    cmd = detect_voice_command("夜間モード オン")

    assert cmd is not None
    assert cmd.action == "night_light_set_enabled"
    assert cmd.args == {"enabled": True}


def test_detect_night_light_strength_set() -> None:
    cmd = detect_voice_command("ナイトライト 45")

    assert cmd is not None
    assert cmd.action == "night_light_strength_set"
    assert cmd.args == {"strength": 45}
