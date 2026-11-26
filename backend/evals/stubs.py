"""
evals/stubs.py - Deterministic stand-ins for the agent's side effects.

* ``StubGeocoder``          fixed gazetteer instead of Nominatim (no network)
* ``InMemoryJobRepository`` job store instead of PostGIS (no database)
* ``FIXED_NOW``             fixed clock, so date ranges are reproducible

Used by the eval runner and by the agent unit tests.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import UUID

from eo_brain.agent.geocoder import GeocodingResult
from eo_brain.agent.validators import clamp_geocoder_bbox
from eo_brain.db.database import JobRecord

FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def normalize(text: Any) -> str:
    """casefold, strip accents, punctuation -> spaces, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(text)).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-word phrase containment on normalised text."""
    h, p = normalize(haystack), normalize(phrase)
    return bool(p) and f" {p} " in f" {h} "


class StubGeocoder:
    """
    Callable with the same contract as ``geocode_or_fail``: returns a
    GeocodingResult or raises ValueError. A query matches an entry when one
    of the entry's aliases (or its full name) appears in the query as whole
    words; the longest matching alias wins. Oversized boxes are clamped the
    same way the real geocoder clamps them.
    """

    def __init__(self, entries: Iterable[Dict[str, Any]]):
        self._entries: List[Dict[str, Any]] = []
        for e in entries:
            bbox = [float(v) for v in e["bbox"]]
            if len(bbox) != 4:
                raise ValueError(f"Gazetteer entry {e.get('name')!r}: bbox needs 4 values")
            aliases = [e["name"], *e.get("aliases", [])]
            self._entries.append({"name": e["name"], "bbox": bbox, "aliases": aliases})
        self.queries: List[str] = []

    def lookup(self, query: str) -> Optional[GeocodingResult]:
        best = None
        best_len = -1
        for entry in self._entries:
            for alias in entry["aliases"]:
                if contains_phrase(query, alias) and len(normalize(alias)) > best_len:
                    best, best_len = entry, len(normalize(alias))
        if best is None:
            return None
        w, s, e, n = clamp_geocoder_bbox(tuple(best["bbox"]))
        return GeocodingResult(
            name=best["name"],
            lat=(s + n) / 2,
            lon=(w + e) / 2,
            bbox=(w, s, e, n),
            source="stub",
        )

    def __call__(self, query: str) -> GeocodingResult:
        self.queries.append(query)
        result = self.lookup(query)
        if result is None:
            raise ValueError(
                f"Could not geocode '{query}'. Try a more specific place name "
                f"(e.g. 'Rome, Italy' instead of 'Roma')."
            )
        return result


class InMemoryJobRepository:
    """The subset of ``eo_brain.db.database.JobRepository`` the agent uses."""

    def __init__(self, now: datetime = FIXED_NOW):
        self._jobs: Dict[UUID, JobRecord] = {}
        self._counter = 0
        self._now = now

    def create(
        self,
        algorithm: str,
        provider: str,
        bbox: Sequence[float],
        start_date: datetime,
        end_date: datetime,
        params: Optional[dict] = None,
    ) -> JobRecord:
        self._counter += 1
        job = JobRecord(
            id=UUID(int=self._counter),  # deterministic ids
            status="pending",
            algorithm=algorithm,
            provider=provider,
            bbox=list(bbox),
            start_date=start_date,
            end_date=end_date,
            params=dict(params or {}),
            progress=0,
            message=None,
            result=None,
            cog_path=None,
            output_crs=None,
            created_at=self._now,
            updated_at=self._now,
            completed_at=None,
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: UUID) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def all(self) -> List[JobRecord]:
        return list(self._jobs.values())


class StubDataAgent:
    """
    Stand-in for the data agent in the tool-routing suite: records the
    question and returns an empty successful result. Routing cases score
    only which tool the orchestrator picked, so no database is needed.
    """

    def __init__(self):
        self.questions: List[str] = []

    def answer(self, question: str, *, area_hint: Optional[str] = None):
        from eo_brain.agent.data_agent import DataAgentResult

        self.questions.append(question)
        return DataAgentResult(status="ok", question=question, sql="(stub: not executed)",
                               columns=[], rows=[], row_count=0)
