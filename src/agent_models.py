"""Data models for autonomous agent results."""
from dataclasses import dataclass, field


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
