import re


def to_business_email(text: str) -> str:
    body = _normalize_body(text)
    if not body:
        return ""

    return (
        "件名: ご連絡\n\n"
        "お世話になっております。\n\n"
        f"{body}\n\n"
        "お忙しいところ恐れ入りますが、ご確認のほどよろしくお願いいたします。\n\n"
        "以上、よろしくお願いいたします。"
    )


def _normalize_body(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    # Remove common conversational fillers at sentence head.
    cleaned = re.sub(r"^(えーと|あの|えっと)[、,\s]*", "", cleaned)

    if cleaned and cleaned[-1] not in "。．.!！？?":
        cleaned += "。"

    # Split by Japanese full stop for readable email lines.
    sentences = [s.strip() for s in re.split(r"(?<=。)", cleaned) if s.strip()]
    if not sentences:
        return cleaned

    return "\n".join(sentences)
