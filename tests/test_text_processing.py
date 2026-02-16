from src.text_processing import ProcessOptions, process_text


def test_process_text_removes_generic_fillers_and_tightens_japanese_spacing() -> None:
    raw = "さあ 最終 文書 に えー スペース が 多く 入っ て いる"
    rules = {"filler_words": [], "habit_patterns": []}

    result = process_text(
        raw,
        rules,
        ProcessOptions(auto_edit=True, remove_fillers=True, remove_habits=False),
    )

    assert result.final_text == "さあ最終文書にスペースが多く入っている。"


def test_process_text_preserves_latin_word_spacing() -> None:
    raw = "Open AI えー model test"
    rules = {"filler_words": [], "habit_patterns": []}

    result = process_text(
        raw,
        rules,
        ProcessOptions(auto_edit=True, remove_fillers=True, remove_habits=False),
    )

    assert result.final_text == "Open AI model test。"
