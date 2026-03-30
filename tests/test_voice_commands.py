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
