import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List


@dataclass
class GateResult:
    accepted: bool
    reason: str = ""


@dataclass
class QualityGate:
    blocked_patterns: List[str] = field(default_factory=list)

    def validate(self, original: str, candidate: str, max_change_ratio: float) -> GateResult:
        if not candidate.strip():
            return GateResult(False, "empty_output")

        ratio = self.change_ratio(original, candidate)
        effective_threshold = max_change_ratio
        if len(original) < 20:
            effective_threshold = max(effective_threshold, 0.65)
        if ratio > effective_threshold:
            return GateResult(False, "change_ratio_exceeded")

        for pattern in self.blocked_patterns:
            if not pattern:
                continue
            if re.search(pattern, candidate):
                return GateResult(False, "blocked_pattern")

        return GateResult(True)

    @staticmethod
    def change_ratio(before: str, after: str) -> float:
        if not before and not after:
            return 0.0
        score = SequenceMatcher(a=before, b=after).ratio()
        return 1.0 - score
