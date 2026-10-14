"""Append-only, hash-chained audit log of risk runs.

Every run stores its full inputs (portfolio, market snapshot, parameters) and
its outputs. That gives three properties regulators and risk managers ask for:

* Tamper evidence: each record carries the SHA-256 of the previous one, so
  editing or deleting any past record breaks `verify()`.
* Idempotency: a run's id is derived from its inputs, so recording the same
  run twice returns the existing record instead of writing a duplicate.
* Reproducibility: `tickrisk.engine.replay()` rebuilds a run from its stored
  inputs and checks the numbers still match.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, exact float repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_id_for(kind: str, inputs: Mapping[str, Any]) -> str:
    """Idempotency key: identical kind and inputs always give the same id."""
    return _sha256(canonical_json({"kind": kind, "inputs": inputs}))[:32]


@dataclass(frozen=True)
class RunRecord:
    seq: int
    run_id: str
    kind: str
    recorded_at: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    prev_hash: str
    hash: str

    def body(self) -> dict:
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "kind": self.kind,
            "recorded_at": self.recorded_at,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "prev_hash": self.prev_hash,
        }

    def to_dict(self) -> dict:
        return {**self.body(), "hash": self.hash}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunRecord:
        return cls(**d)


class IntegrityError(Exception):
    """Raised when the hash chain has been broken."""


class EventLog:
    """Hash-chained run log, in memory or backed by a JSON Lines file.

    With a path, every record is appended and fsynced, and the existing file
    is loaded and verified on open.
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: list[RunRecord] = []
        self._by_id: dict[str, RunRecord] = {}
        if self._path and self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self._index(RunRecord.from_dict(json.loads(line)))
            self.verify()

    def _index(self, rec: RunRecord) -> None:
        self._records.append(rec)
        self._by_id[rec.run_id] = rec

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self._records)

    @property
    def head(self) -> str:
        return self._records[-1].hash if self._records else GENESIS

    def get(self, run_id: str) -> RunRecord:
        try:
            return self._by_id[run_id]
        except KeyError:
            raise KeyError(f"No run with id {run_id!r}") from None

    def record(self, kind: str, inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> RunRecord:
        """Append a run. If an identical run already exists, return it unchanged."""
        run_id = run_id_for(kind, inputs)
        if run_id in self._by_id:
            return self._by_id[run_id]

        # Round-trip through JSON so stored data is exactly what replay will read.
        inputs = json.loads(canonical_json(inputs))
        outputs = json.loads(canonical_json(outputs))
        draft = RunRecord(
            seq=len(self._records),
            run_id=run_id,
            kind=kind,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            inputs=inputs,
            outputs=outputs,
            prev_hash=self.head,
            hash="",
        )
        rec = RunRecord(**{**draft.body(), "hash": _sha256(canonical_json(draft.body()))})

        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(canonical_json(rec.to_dict()) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        self._index(rec)
        return rec

    def verify(self) -> None:
        """Check every record's hash and link. Raises IntegrityError on the first break."""
        prev = GENESIS
        for i, rec in enumerate(self._records):
            if rec.seq != i:
                raise IntegrityError(f"record {i} has seq {rec.seq}")
            if rec.prev_hash != prev:
                raise IntegrityError(f"record {i} does not link to the previous record")
            if _sha256(canonical_json(rec.body())) != rec.hash:
                raise IntegrityError(f"record {i} ({rec.run_id}) has been modified")
            if run_id_for(rec.kind, rec.inputs) != rec.run_id:
                raise IntegrityError(f"record {i} inputs do not match its run id")
            prev = rec.hash
