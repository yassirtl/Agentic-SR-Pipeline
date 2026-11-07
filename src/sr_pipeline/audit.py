"""Audit trail on disk: runs/<run_id>/audit.jsonl (one JSON object per line).

The audit log lives in the graph state (append-only list). ``AuditWriter.sync``
copies the entries that are not on disk yet, so it can be called after every
invoke / resume without duplicating lines. JSONL is append-only and stays
readable if a run stops halfway.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def new_run_id() -> str:
    """Sortable, unique run id, e.g. 20261107-142530-a1b2c3."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditWriter:
    def __init__(self, runs_dir: str | Path, run_id: str):
        self.run_id = run_id
        self.run_dir = Path(runs_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "audit.jsonl"
        # Re-opening an existing run (resume): count what is already written.
        self.written = sum(1 for _ in self.path.open()) if self.path.exists() else 0

    def sync(self, audit_log: list[dict]) -> int:
        """Append entries of ``audit_log`` not written yet. Returns how many were written."""
        new = audit_log[self.written:]
        if new:
            with self.path.open("a") as f:
                for entry in new:
                    f.write(json.dumps({"run_id": self.run_id, "written_at": _utc_now(),
                                        **entry}, default=str) + "\n")
            self.written += len(new)
        return len(new)

    def save_state(self, state: dict) -> Path:
        """Snapshot of the full state (for inspection; the checkpointer is the real store)."""
        path = self.run_dir / "state.json"
        path.write_text(json.dumps(state, indent=2, default=str))
        return path


def read_audit(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
