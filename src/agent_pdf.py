"""PDF extraction, file download helpers, and Excel writer for autonomous agent."""
import csv
import re
import zipfile
from pathlib import Path
from urllib import parse as urllib_parse
from xml.sax.saxutils import escape as xml_escape


def extract_text_from_pdf_bytes(raw: bytes) -> str:
    chunks: list[str] = []
    for match in re.finditer(rb"\(([^()]*)\)\s*Tj", raw):
        try:
            chunks.append(match.group(1).decode("utf-8"))
        except UnicodeDecodeError:
            chunks.append(match.group(1).decode("latin-1", errors="ignore"))
    for match in re.finditer(rb"<([0-9A-Fa-f]+)>\s*Tj", raw):
        hex_part = match.group(1)
        try:
            chunks.append(bytes.fromhex(hex_part.decode("ascii")).decode("utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            continue
    if not chunks:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    return "\n".join(chunks)


def write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Summary" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""
    row_xml: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            col_name = _excel_col_name(col_idx)
            text = xml_escape(str(value))
            cells.append(f'<c r="{col_name}{row_idx}" t="inlineStr"><is><t>{text}</t></is></c>')
        row_xml.append(f'<row r="{row_idx}">' + "".join(cells) + "</row>")
    sheet = (
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"""
        """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"""
        """<sheetData>"""
        + "".join(row_xml)
        + """</sheetData></worksheet>"""
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


def write_csv_fallback(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)


def _excel_col_name(index: int) -> str:
    label = ""
    n = index
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label = chr(65 + rem) + label
    return label


def extract_paths(goal: str) -> list[Path]:
    values: list[str] = []
    values.extend(re.findall(r'["\']([^"\']+)["\']', goal))
    values.extend(re.findall(r'["\']([A-Za-z]:\\[^"\']+)["\']', goal))
    values.extend(re.findall(r"\b([A-Za-z]:\\[^\s,，。]+)", goal))
    values.extend(re.findall(r'["\'](/[^"\']+)["\']', goal))

    paths: list[Path] = []
    for value in values:
        raw = value.strip()
        if "://" in raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.exists():
            paths.append(candidate)
    return dedupe_paths(paths)


def extract_urls(goal: str) -> list[str]:
    urls = re.findall(r"https?://[^\s'\"<>]+", goal)
    deduped = []
    for url in urls:
        normalized = url.strip().rstrip(".,)")
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def has_download_hint(goal: str) -> bool:
    goal_lower = goal.lower()
    return "download" in goal_lower or "ダウンロード" in goal


def dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def safe_pdf_name(url: str, index: int) -> str:
    parsed = urllib_parse.urlparse(url)
    name = Path(parsed.path).name or f"download_{index + 1}.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not safe:
        safe = f"download_{index + 1}.pdf"
    return safe
