"""
evals/stubs.py - Deterministic stand-ins for the agent's side effects.

* ``StubGeocoder``          recorded Nominatim responses or a fixed gazetteer (no network),
                            with the real candidate selection on top
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

from eo_brain.agent.geocoder import GeocodingResult, PlaceResolver
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
    GeocodingResult, or raises PlaceNotFoundError / PlaceChoiceNeeded (both
    ValueError). The real candidate selection (``geocoder.PlaceResolver``)
    runs on top of a stub search, so the evals exercise it too:

    * the gazetteer first: a search matches an entry when one of the entry's
      aliases (or its full name) appears in the text as whole words; the
      longest matching alias wins. The entry is returned as one
      administrative-area candidate whose names are its aliases. (Searched
      first so that the cases written against it keep their places and boxes.)
    * otherwise ``recorded``: real Nominatim responses keyed by the searched
      text (``evals/fixtures/nominatim_recorded.json``), matched ignoring case
      and spacing but not punctuation, as the production cache does.

    Oversized boxes are clamped the same way the real geocoder clamps them.
    """

    def __init__(self, entries: Iterable[Dict[str, Any]], recorded: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self._entries: List[Dict[str, Any]] = []
        for e in entries:
            bbox = [float(v) for v in e["bbox"]]
            if len(bbox) != 4:
                raise ValueError(f"Gazetteer entry {e.get('name')!r}: bbox needs 4 values")
            aliases = [e["name"], *e.get("aliases", [])]
            self._entries.append({"name": e["name"], "bbox": bbox, "aliases": aliases,
                                  "kind": e.get("kind", "city")})
        self._recorded = {_search_key(k): list(v) for k, v in (recorded or {}).items()}
        self._resolver = PlaceResolver(self.search)
        self.queries: List[str] = []    # what the agent asked to geocode
        self.searches: List[str] = []   # what the resolver searched for

    def _entry(self, text: str) -> Optional[Dict[str, Any]]:
        best, best_len = None, -1
        for entry in self._entries:
            for alias in entry["aliases"]:
                if contains_phrase(text, alias) and len(normalize(alias)) > best_len:
                    best, best_len = entry, len(normalize(alias))
        return best

    def search(self, text: str) -> List[Dict[str, Any]]:
        self.searches.append(text)
        entry = self._entry(text)
        if entry is None:
            return self._recorded.get(_search_key(text), [])
        w, s, e, n = entry["bbox"]
        short = entry["name"].split(",")[0].strip()
        return [{
            "class": "boundary", "type": "administrative", "addresstype": entry["kind"],
            "place_rank": 16, "importance": 0.5,
            "name": short, "display_name": entry["name"],
            "namedetails": {"name": short, "alt_name": ";".join(entry["aliases"])},
            "lat": str((s + n) / 2), "lon": str((w + e) / 2),
            "boundingbox": [str(s), str(n), str(w), str(e)],
        }]

    def __call__(self, query: str) -> GeocodingResult:
        self.queries.append(query)
        return self._resolver.resolve(query)


def _search_key(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def load_recorded_searches(path) -> Dict[str, List[Dict[str, Any]]]:
    """The ``responses`` of a recorded-searches fixture (see evals/fixtures/nominatim_recorded.json)."""
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))["responses"]


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
