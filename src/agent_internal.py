"""Internal autonomous agent: local task execution (PDF, Excel, email)."""
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib import request as urllib_request

from .agent_models import AgentStep, AutonomousAgentResult
from .agent_pdf import (
    dedupe_paths,
    extract_paths,
    extract_text_from_pdf_bytes,
    extract_urls,
    has_download_hint,
    safe_pdf_name,
    write_csv_fallback,
    write_minimal_xlsx,
)


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
        source_urls = extract_urls(goal)
        if source_dirs or source_urls:
            steps.append(AgentStep(
                name="resolve_inputs",
                status="completed",
                detail=f"dirs={len(source_dirs)}, urls={len(source_urls)}",
            ))

        if needs_pdf:
            ok, detail, attempts = self._with_retries(
                lambda: self._discover_pdf_files(goal=goal, run_dir=run_dir),
                "no_pdf_found",
            )
            if ok:
                pdf_files = detail  # type: ignore[assignment]
                steps.append(AgentStep(
                    name="collect_pdf_files",
                    status="completed",
                    detail=f"{len(pdf_files)} file(s) found",
                    attempts=attempts,
                ))
            else:
                success = False
                steps.append(AgentStep(
                    name="collect_pdf_files",
                    status="failed",
                    detail=str(detail),
                    attempts=attempts,
                ))

        if needs_pdf and pdf_files:
            for pdf_path in pdf_files:
                ok, detail, attempts = self._with_retries(
                    lambda p=pdf_path: self._summarize_pdf(p), "pdf_parse_failed"
                )
                if ok:
                    summaries.append((pdf_path.name, str(detail)))
                    steps.append(AgentStep(
                        name="summarize_pdf",
                        status="completed",
                        detail=pdf_path.name,
                        attempts=attempts,
                    ))
                    continue

                fallback = f"Failed to parse body. Filename-based summary: {pdf_path.name}"
                fixes.append(f"PDF parse failed -> filename summary ({pdf_path.name})")
                summaries.append((pdf_path.name, fallback))
                steps.append(AgentStep(
                    name="summarize_pdf",
                    status="repaired",
                    detail=fallback,
                    attempts=attempts,
                ))

        if needs_excel:
            rows = [["File", "Summary"]]
            for name, summary in summaries:
                rows.append([name, summary])
            if len(rows) == 1:
                rows.append(["N/A", "No PDF summary target found."])

            xlsx_path = run_dir / "summary.xlsx"
            try:
                write_minimal_xlsx(xlsx_path, rows)
                artifact_paths.append(str(xlsx_path))
                steps.append(AgentStep(
                    name="export_excel",
                    status="completed",
                    detail="summary.xlsx",
                    output_path=str(xlsx_path),
                ))
            except Exception as exc:  # noqa: BLE001
                csv_path = run_dir / "summary.csv"
                write_csv_fallback(csv_path, rows)
                artifact_paths.append(str(csv_path))
                fixes.append("Excel export failed -> CSV fallback")
                steps.append(AgentStep(
                    name="export_excel",
                    status="repaired",
                    detail=f"xlsx failed ({type(exc).__name__}), fallback to summary.csv",
                    output_path=str(csv_path),
                ))

        if needs_mail:
            email_path = run_dir / "mail_template.txt"
            body = self._build_email_template(goal=goal, summaries=summaries)
            email_path.write_text(body, encoding="utf-8")
            artifact_paths.append(str(email_path))
            steps.append(AgentStep(
                name="build_mail_template",
                status="completed",
                detail="mail_template.txt",
                output_path=str(email_path),
            ))

        report_path = run_dir / "agent_report.md"
        report_text = self._build_report(
            goal=goal, started=started, steps=steps,
            artifact_paths=artifact_paths, fixes=fixes,
        )
        report_path.write_text(report_text, encoding="utf-8")
        artifact_paths.append(str(report_path))

        summary = "Autonomous run completed" if success else "Autonomous run completed with failures"
        return AutonomousAgentResult(
            goal=goal, mode="internal", success=success, summary=summary,
            report_path=str(report_path), artifact_paths=artifact_paths, steps=steps,
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
        source_urls = extract_urls(goal)
        files: list[Path] = []
        files.extend(self._discover_local_pdf_files(source_dirs))
        if source_urls:
            files.extend(self._download_pdf_files_from_urls(source_urls, run_dir))
        files = dedupe_paths(files)
        if not files:
            detail = f"no_pdf_found (searched_dirs={len(source_dirs)}, urls={len(source_urls)})"
            raise RuntimeError(detail)
        return files

    def _resolve_source_dirs(self, goal: str) -> list[Path]:
        dirs: list[Path] = [self.workspace_root]
        for path in extract_paths(goal):
            if path.is_file() and path.suffix.lower() == ".pdf":
                dirs.append(path.parent)
            elif path.is_dir():
                dirs.append(path)
        if has_download_hint(goal):
            downloads = Path.home() / "Downloads"
            if downloads.exists() and downloads.is_dir():
                dirs.append(downloads)
        return dedupe_paths(dirs)

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
        from urllib import parse as urllib_parse
        out_dir = run_dir / "downloaded_pdfs"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for url in urls[:10]:
            parsed = urllib_parse.urlparse(url)
            if parsed.path.lower().endswith(".pdf"):
                target = out_dir / safe_pdf_name(url, len(saved))
                self._download_file(url=url, destination=target)
                saved.append(target)
                continue

            candidates = self._extract_pdf_links_from_page(url)
            for candidate_url in candidates[:20]:
                target = out_dir / safe_pdf_name(candidate_url, len(saved))
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
        import re
        from urllib import parse as urllib_parse
        request = urllib_request.Request(url=url, method="GET")
        with urllib_request.urlopen(request, timeout=20.0) as response:
            html = response.read().decode("utf-8", errors="replace")
        links = re.findall(r"""href=["']([^"']+?\.pdf(?:\?[^"']*)?)["']""", html, flags=re.IGNORECASE)
        resolved = [urllib_parse.urljoin(url, link.strip()) for link in links if link.strip()]
        return list(dict.fromkeys(resolved))

    def _summarize_pdf(self, path: Path) -> str:
        text = extract_text_from_pdf_bytes(path.read_bytes())
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
            "# Autonomous Agent Report", "",
            f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
            "- mode: internal",
            f"- goal: {goal}",
            f"- elapsed_ms: {elapsed_ms}", "",
            "## Steps",
        ]
        for step in steps:
            lines.append(f"- {step.name}: {step.status} (attempts={step.attempts}) {step.detail}".rstrip())
        lines += ["", "## Artifacts"]
        for path in artifact_paths:
            lines.append(f"- {path}")
        lines += ["", "## Self-repair"]
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
