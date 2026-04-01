"""Autonomous agent facade.

InternalAutonomousAgent  -> agent_internal.py (local task execution)
ExternalAPIAutonomousAgent -> this file (external API delegation)

Re-exports InternalAutonomousAgent and data models for backwards compatibility.
"""
import json
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from .agent_models import AgentStep, AutonomousAgentResult
from .agent_internal import InternalAutonomousAgent

__all__ = [
    "AgentStep",
    "AutonomousAgentResult",
    "InternalAutonomousAgent",
    "ExternalAPIAutonomousAgent",
]


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
                steps.append(AgentStep(
                    name=str(item.get("name", "external_step")),
                    status=str(item.get("status", "completed")),
                    detail=str(item.get("detail", "")),
                    attempts=int(item.get("attempts", 1)),
                    output_path=str(item.get("output_path", "")),
                ))
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
