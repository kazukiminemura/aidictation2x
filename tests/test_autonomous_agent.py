import json
import shutil
import uuid
from pathlib import Path

from src.autonomous_agent import ExternalAPIAutonomousAgent, InternalAutonomousAgent


def _make_workspace() -> Path:
    root = Path("data") / "tmp-tests"
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / f"agent-test-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def test_internal_agent_generates_excel_mail_and_report() -> None:
    tmp_path = _make_workspace()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nBT\n(Hello PDF summary text) Tj\nET\n")

    agent = InternalAutonomousAgent(workspace_root=tmp_path)
    result = agent.run("このフォルダのPDFを全部読み込んで、要点をまとめてExcelに書き出して、最後にメール文面のテンプレを作って")

    try:
        assert result.success is True
        assert result.mode == "internal"
        assert any(path.endswith("summary.xlsx") for path in result.artifact_paths)
        assert any(path.endswith("mail_template.txt") for path in result.artifact_paths)
        assert result.report_path.endswith("agent_report.md")
        assert Path(result.report_path).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_external_api_agent_parses_json_response() -> None:
    tmp_path = _make_workspace()

    def fake_caller(url: str, payload: dict, timeout_ms: int) -> tuple[str, str]:  # noqa: ARG001
        response = {
            "summary": "External completed",
            "report_path": str(tmp_path / "external-report.md"),
            "artifact_paths": [str(tmp_path / "summary.xlsx")],
            "steps": [{"name": "plan", "status": "completed", "detail": "ok"}],
        }
        raw = json.dumps(response, ensure_ascii=False)
        return "External completed", raw

    agent = ExternalAPIAutonomousAgent(
        endpoint_url="http://127.0.0.1:8000/v1/agent/run",
        caller=fake_caller,
    )
    result = agent.run("run external autonomous task", workspace_root=tmp_path)

    try:
        assert result.mode == "external_api"
        assert result.success is True
        assert result.summary == "External completed"
        assert len(result.steps) == 1
        assert result.steps[0].name == "plan"
        assert result.artifact_paths[0].endswith("summary.xlsx")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_internal_agent_uses_path_in_goal() -> None:
    tmp_path = _make_workspace()
    source_dir = tmp_path / "source_docs"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "outside.pdf").write_bytes(b"%PDF-1.4\nBT\n(Outside source PDF) Tj\nET\n")

    goal = f'PDFを読み込んでExcelにまとめて。対象フォルダ: "{source_dir}"'
    agent = InternalAutonomousAgent(workspace_root=tmp_path / "another_root")
    result = agent.run(goal)

    try:
        assert result.success is True
        assert any(step.name == "collect_pdf_files" and step.status == "completed" for step in result.steps)
        assert any(path.endswith("summary.xlsx") for path in result.artifact_paths)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
