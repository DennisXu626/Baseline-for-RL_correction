"""Small serializable curriculum state matching the pinned ViViDex trigger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class CurriculumState:
    stage: int = 0
    evaluations: int = 0
    transitions: int = 0
    threshold: float = 0.95
    internal_episodes: int = 25

    def update(self, pregrasp_successes: list[float]) -> bool:
        if len(pregrasp_successes) != self.internal_episodes:
            raise ValueError(
                f"internal curriculum evaluation requires exactly {self.internal_episodes} episodes"
            )
        self.evaluations += 1
        mean = sum(float(value) for value in pregrasp_successes) / len(pregrasp_successes)
        if mean > self.threshold and self.stage <= 1:
            self.stage += 1
            self.transitions += 1
            return True
        return False

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CurriculumState":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

