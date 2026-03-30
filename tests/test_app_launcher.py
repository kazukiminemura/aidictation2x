from src.app_launcher import AppLauncher


def test_browser_search_opens_google_query(monkeypatch) -> None:
    opened: list[str] = []

    def fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr("webbrowser.open", fake_open)

    label = AppLauncher().browser_search("渋谷 ランチ")

    assert label == "ブラウザ"
    assert opened == ["https://www.google.com/search?q=%E6%B8%8B%E8%B0%B7+%E3%83%A9%E3%83%B3%E3%83%81"]


def test_browser_search_rejects_empty_query() -> None:
    launcher = AppLauncher()

    try:
        launcher.browser_search("   ")
    except ValueError as exc:
        assert str(exc) == "検索語が空です"
    else:
        raise AssertionError("ValueError was not raised")


def test_browser_open_result_opens_lucky_query(monkeypatch) -> None:
    opened: list[str] = []

    def fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr("webbrowser.open", fake_open)

    label = AppLauncher().browser_open_result("OpenAI")

    assert label == "ブラウザ"
    assert opened == ["https://www.google.com/search?btnI=I&q=OpenAI"]


def test_browser_back_uses_back_shortcut(monkeypatch) -> None:
    calls: list[str] = []
    launcher = AppLauncher()
    monkeypatch.setattr(launcher, "_send_browser_shortcut", calls.append)

    label = launcher.browser_back()

    assert label == "ブラウザ"
    assert calls == ["back"]


def test_browser_forward_uses_forward_shortcut(monkeypatch) -> None:
    calls: list[str] = []
    launcher = AppLauncher()
    monkeypatch.setattr(launcher, "_send_browser_shortcut", calls.append)

    label = launcher.browser_forward()

    assert label == "ブラウザ"
    assert calls == ["forward"]


def test_browser_open_url_adds_https(monkeypatch) -> None:
    opened: list[str] = []

    def fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr("webbrowser.open", fake_open)

    label = AppLauncher().browser_open_url("example.com")

    assert label == "ブラウザ"
    assert opened == ["https://example.com"]


def test_browser_open_url_rejects_invalid_target() -> None:
    launcher = AppLauncher()

    try:
        launcher.browser_open_url("bad target")
    except ValueError as exc:
        assert str(exc) == "リンク先の形式が不正です"
    else:
        raise AssertionError("ValueError was not raised")
