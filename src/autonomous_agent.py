import csv
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from xml.sax.saxutils import escape as xml_escape


@dataclass
class AgentStep:
    name: str
    status: str
    detail: str = ""
    attempts: int = 1
    output_path: str = ""


@dataclass
class AutonomousAgentResult:
    goal: str
    mode: str
    success: bool
    summary: str
    report_path: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    external_raw_response: str = ""


class InternalAutonomousAgent:
    def __init__(self, workspace_root: Path, max_retries: int = 2):
        self.workspace_root = workspace_root
        self.max_retries = max(1, max_retries)

    def run(self, goal: str) -> AutonomousAgentResult:
        started = time.perf_counter()
        run_dir = self._new_run_dir(self.workspace_root)
        steps: list[AgentStep] = []
        artifact_paths: list[str] = []
        summaries: list[tuple[str, str]] = []
        fixes: list[str] = []
        pdf_files: list[Path] = []
        success = True

        goal_lower = goal.lower()
        needs_pdf = "pdf" in goal_lower
        needs_excel = "excel" in goal_lower or "xlsx" in goal_lower
        needs_mail = ("mail" in goal_lower) or ("email" in goal_lower) or ("メール" in goal)

        source_dirs = self._resolve_source_dirs(goal)
        source_urls = _extract_urls(goal)
        if source_dirs or source_urls:
            steps.append(
                AgentStep(
                    name="resolve_inputs",
                    status="completed",
                    detail=(
                        f"dirs={len(source_dirs)}, urls={len(source_urls)}"
                    ),
                )
            )

        if needs_pdf:
            ok, detail, attempts = self._with_retries(
                lambda: self._discover_pdf_files(goal=goal, run_dir=run_dir),
                "no_pdf_found",
            )
            if ok:
                pdf_files = detail  # type: ignore[assignment]
                steps.append(
                    AgentStep(
                        name="collect_pdf_files",
                        status="completed",
                        detail=f"{len(pdf_files)} file(s) found",
                        attempts=attempts,
                    )
                )
            else:
                success = False
                steps.append(
                    AgentStep(
                        name="collect_pdf_files",
                        status="failed",
                        detail=str(detail),
                        attempts=attempts,
                    )
                )

        if needs_pdf and pdf_files:
            for pdf_path in pdf_files:
                ok, detail, attempts = self._with_retries(lambda p=pdf_path: self._summarize_pdf(p), "pdf_parse_failed")
                if ok:
                    summaries.append((pdf_path.name, str(detail)))
                    steps.append(
                        AgentStep(
                            name="summarize_pdf",
                            status="completed",
                            detail=pdf_path.name,
                            attempts=attempts,
                        )
                    )
                    continue

                fallback = f"Failed to parse body. Filename-based summary: {pdf_path.name}"
                fixes.append(f"PDF parse failed -> filename summary ({pdf_path.name})")
                summaries.append((pdf_path.name, fallback))
                steps.append(
                    AgentStep(
                        name="summarize_pdf",
                        status="repaired",
                        detail=fallback,
                        attempts=attempts,
                    )
                )

        if needs_excel:
            rows = [["File", "Summary"]]
            for name, summary in summaries:
                rows.append([name, summary])
            if len(rows) == 1:
                rows.append(["N/A", "No PDF summary target found."])

            xlsx_path = run_dir / "summary.xlsx"
            try:
                _write_minimal_xlsx(xlsx_path, rows)
                artifact_paths.append(str(xlsx_path))
                steps.append(
                    AgentStep(
                        name="export_excel",
                        status="completed",
                        detail="summary.xlsx",
                        output_path=str(xlsx_path),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                csv_path = run_dir / "summary.csv"
                with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
                    writer = csv.writer(fp)
                    writer.writerows(rows)
                artifact_paths.append(str(csv_path))
                fixes.append("Excel export failed -> CSV fallback")
                steps.append(
                    AgentStep(
                        name="export_excel",
                        status="repaired",
                        detail=f"xlsx failed ({type(exc).__name__}), fallback to summary.csv",
                        output_path=str(csv_path),
                    )
                )

        if needs_mail:
            email_path = run_dir / "mail_template.txt"
            body = self._build_email_template(goal=goal, summaries=summaries)
            email_path.write_text(body, encoding="utf-8")
            artifact_paths.append(str(email_path))
            steps.append(
                AgentStep(
                    name="build_mail_template",
                    status="completed",
                    detail="mail_template.txt",
                    output_path=str(email_path),
                )
            )

        report_path = run_dir / "agent_report.md"
        report_text = self._build_report(
            goal=goal,
            started=started,
            steps=steps,
            artifact_paths=artifact_paths,
            fixes=fixes,
        )
        report_path.write_text(report_text, encoding="utf-8")
        artifact_paths.append(str(report_path))

        summary = "Autonomous run completed" if success else "Autonomous run completed with failures"
        return AutonomousAgentResult(
            goal=goal,
            mode="internal",
            success=success,
            summary=summary,
            report_path=str(report_path),
            artifact_paths=artifact_paths,
            steps=steps,
        )

    def _with_retries(self, fn: Callable[[], Any], default_error: str) -> tuple[bool, Any, int]:
        last_error = default_error
        for attempt in range(1, self.max_retries + 1):
            try:
                return True, fn(), attempt
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc).strip() or default_error
        return False, last_error, self.max_retries

    def _discover_pdf_files(self, goal: str, run_dir: Path) -> list[Path]:
        source_dirs = self._resolve_source_dirs(goal)
        source_urls = _extract_urls(goal)
        files: list[Path] = []
        files.extend(self._discover_local_pdf_files(source_dirs))
        if source_urls:
            files.extend(self._download_pdf_files_from_urls(source_urls, run_dir))
        files = _dedupe_paths(files)
        if not files:
            detail = (
                f"no_pdf_found (searched_dirs={len(source_dirs)}, urls={len(source_urls)})"
            )
            raise RuntimeError(detail)
        return files

    def _resolve_source_dirs(self, goal: str) -> list[Path]:
        dirs: list[Path] = [self.workspace_root]
        for path in _extract_paths(goal):
            if path.is_file() and path.suffix.lower() == ".pdf":
                dirs.append(path.parent)
            elif path.is_dir():
                dirs.append(path)
        if _has_download_hint(goal):
            downloads = Path.home() / "Downloads"
            if downloads.exists() and downloads.is_dir():
                dirs.append(downloads)
        return _dedupe_paths(dirs)

    def _discover_local_pdf_files(self, source_dirs: list[Path]) -> list[Path]:
        files: list[Path] = []
        for source_dir in source_dirs:
            if not source_dir.exists() or not source_dir.is_dir():
                continue
            for path in source_dir.rglob("*.pdf"):
                parts = {part.lower() for part in path.parts}
                if {"venv", ".git", "dist", "build", "__pycache__"} & parts:
                    continue
                files.append(path)
        return files

    def _download_pdf_files_from_urls(self, urls: list[str], run_dir: Path) -> list[Path]:
        out_dir = run_dir / "downloaded_pdfs"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for url in urls[:10]:
            parsed = urllib_parse.urlparse(url)
            if parsed.path.lower().endswith(".pdf"):
                target = out_dir / _safe_pdf_name(url, len(saved))
                self._download_file(url=url, destination=target)
                saved.append(target)
                continue

            candidates = self._extract_pdf_links_from_page(url)
            for candidate_url in candidates[:20]:
                target = out_dir / _safe_pdf_name(candidate_url, len(saved))
                self._download_file(url=candidate_url, destination=target)
                saved.append(target)
        return saved

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        request = urllib_request.Request(url=url, method="GET")
        with urllib_request.urlopen(request, timeout=20.0) as response:
            payload = response.read()
        if not payload:
            raise RuntimeError("empty_download")
        destination.write_bytes(payload)

    @staticmethod
    def _extract_pdf_links_from_page(url: str) -> list[str]:
        request = urllib_request.Request(url=url, method="GET")
        with urllib_request.urlopen(request, timeout=20.0) as response:
            html = response.read().decode("utf-8", errors="replace")
        links = re.findall(r"""href=["']([^"']+?\.pdf(?:\?[^"']*)?)["']""", html, flags=re.IGNORECASE)
        resolved = [urllib_parse.urljoin(url, link.strip()) for link in links if link.strip()]
        return list(dict.fromkeys(resolved))

    def _summarize_pdf(self, path: Path) -> str:
        text = _extract_text_from_pdf_bytes(path.read_bytes())
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("pdf_parse_failed")
        joined = " ".join(lines)
        if len(joined) > 220:
            joined = joined[:220].rstrip() + "..."
        return joined

    @staticmethod
    def _build_email_template(goal: str, summaries: list[tuple[str, str]]) -> str:
        bullet_lines = [f"- {name}: {summary}" for name, summary in summaries[:10]]
        if not bullet_lines:
            bullet_lines.append("- No summary target found.")
        return (
            "Subject: Shared summary report\n\n"
            "Hello team,\n\n"
            "Please find the autonomous run result below.\n\n"
            f"Goal:\n{goal}\n\n"
            "Highlights:\n"
            + "\n".join(bullet_lines)
            + "\n\nBest regards,\n"
        )

    @staticmethod
    def _build_report(
        goal: str,
        started: float,
        steps: list[AgentStep],
        artifact_paths: list[str],
        fixes: list[str],
    ) -> str:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        lines = [
            "# Autonomous Agent Report",
            "",
            f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
            "- mode: internal",
            f"- goal: {goal}",
            f"- elapsed_ms: {elapsed_ms}",
            "",
            "## Steps",
        ]
        for step in steps:
            lines.append(f"- {step.name}: {step.status} (attempts={step.attempts}) {step.detail}".rstrip())
        lines.append("")
        lines.append("## Artifacts")
        for path in artifact_paths:
            lines.append(f"- {path}")
        lines.append("")
        lines.append("## Self-repair")
        if fixes:
            for fix in fixes:
                lines.append(f"- {fix}")
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _new_run_dir(workspace_root: Path) -> Path:
        root = workspace_root / "data" / "agent_runs"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = root / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


class ExternalAPIAutonomousAgent:
    def __init__(
        self,
        endpoint_url: str,
        timeout_ms: int = 300000,
        caller: Callable[[str, dict[str, Any], int], tuple[str, str]] | None = None,
    ):
        self.endpoint_url = endpoint_url
        self.timeout_ms = max(1000, timeout_ms)
        self.caller = caller or _call_external_api

    def run(self, goal: str, workspace_root: Path) -> AutonomousAgentResult:
        prompt = {
            "goal": goal,
            "workspace": str(workspace_root),
            "instructions": (
                "Decompose and execute tasks. Return JSON with summary, report_path, artifact_paths, and steps."
            ),
        }
        text, raw = self.caller(self.endpoint_url, prompt, self.timeout_ms)

        payload: dict[str, Any] = {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}

        steps: list[AgentStep] = []
        raw_steps = payload.get("steps")
        if isinstance(raw_steps, list):
            for item in raw_steps:
                if not isinstance(item, dict):
                    continue
                steps.append(
                    AgentStep(
                        name=str(item.get("name", "external_step")),
                        status=str(item.get("status", "completed")),
                        detail=str(item.get("detail", "")),
                        attempts=int(item.get("attempts", 1)),
                        output_path=str(item.get("output_path", "")),
                    )
                )
        if not steps:
            steps = [AgentStep(name="external_agent_call", status="completed", detail="response received")]

        artifact_paths: list[str] = []
        raw_artifacts = payload.get("artifact_paths")
        if isinstance(raw_artifacts, list):
            artifact_paths = [str(item) for item in raw_artifacts if str(item).strip()]

        return AutonomousAgentResult(
            goal=goal,
            mode="external_api",
            success=True,
            summary=str(payload.get("summary", text or "External agent completed")),
            report_path=str(payload.get("report_path", "")),
            artifact_paths=artifact_paths,
            steps=steps,
            external_raw_response=raw,
        )


def _extract_text_from_pdf_bytes(raw: bytes) -> str:
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


def _write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
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


def _excel_col_name(index: int) -> str:
    label = ""
    n = index
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label = chr(65 + rem) + label
    return label


def _call_external_api(url: str, payload: dict[str, Any], timeout_ms: int) -> tuple[str, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=max(1.0, timeout_ms / 1000.0)) as response:
            raw = response.read().decode("utf-8", errors="replace").strip()
    except (TimeoutError, urllib_error.HTTPError, urllib_error.URLError) as exc:
        raise RuntimeError("external_autonomous_agent_error") from exc

    if not raw:
        raise RuntimeError("external_autonomous_agent_bad_response")
    try:
        payload_json = json.loads(raw)
    except json.JSONDecodeError:
        return raw, raw
    text = _extract_text(payload_json)
    if not text:
        text = raw
    return text, raw


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "summary", "response", "content", "message", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            nested = _extract_text(value)
            if nested:
                return nested
    if isinstance(payload, list):
        for item in payload:
            nested = _extract_text(item)
            if nested:
                return nested
    return ""


def _extract_paths(goal: str) -> list[Path]:
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
    return _dedupe_paths(paths)


def _extract_urls(goal: str) -> list[str]:
    urls = re.findall(r"https?://[^\s'\"<>]+", goal)
    deduped = []
    for url in urls:
        normalized = url.strip().rstrip(".,)")
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _has_download_hint(goal: str) -> bool:
    goal_lower = goal.lower()
    return "download" in goal_lower or "ダウンロード" in goal


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _safe_pdf_name(url: str, index: int) -> str:
    parsed = urllib_parse.urlparse(url)
    name = Path(parsed.path).name or f"download_{index + 1}.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not safe:
        safe = f"download_{index + 1}.pdf"
    return safe
