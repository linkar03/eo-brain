"""
eo_brain/agent/geocoder.py - Place name to analysis area (Nominatim / OSM).

An analysis needs an AREA, and the user's text names one in many ways
("Casentino, Arezzo, Italia", "casentino arezzo, italia", "vallata del
Casentino"). Nominatim matches names, not intentions, and its first result is
often not the area meant: a street whose address contains the words, a
village with the same name in another region. So the search asks for several
candidates and code chooses among them, by these rules:

1. Only areas compete. A candidate is an area when its OSM class is a
   boundary, place, natural feature, waterway, water body, land use or leisure
   area (not a street, building, shop, attraction or single tree) and its
   bounding box passes the same minimum-extent check the analysis applies
   (``validators.validate_bbox``). A non-area is used only if no area matches
   at all, its own box (not padding around a point) passes that check, and
   the result says what it is.
2. The name must match. The searched name (the first comma-separated part,
   without a leading article or generic descriptor) must equal one of the
   candidate's names (``namedetails``: name, name:xx, alt/official/old names;
   never ``short_name`` or ``ref``, where codes such as "PO" for Prato live), or
   one must contain the other word for word. A candidate that matches only
   through other words (search "Po", result "Prato") is a reinterpretation by
   the search engine and is never taken.
3. A stripped descriptor states the kind. "vallata / valle / valley / regione /
   region / territorio / comprensorio / piana / pianura" ask for a region, so a
   settlement does not qualify; "provincia / province" asks for a province.
   When no named candidate has the stated kind, the user is asked (a region was
   asked for and only a village found): the kind is never changed silently.
   "zona / area / dintorni / around / near ..." state no kind.
4. Genuinely ambiguous means: two or more candidates have exactly the searched
   name, are different places (neither one's centre lies inside the other's
   box), and the runner-up's Nominatim importance is at least AMBIGUITY_RATIO
   (90 %) of the best one's, after rules 1-3. Nothing in the text separates
   them, so the user is asked (``PlaceChoiceNeeded`` lists them, with kind and
   province/region). Otherwise the exact-name candidate with the highest
   importance wins. Nested places of one name (Florence the city and Florence
   the metropolitan city) are one place, not a choice.
5. Punctuation does not decide. When the text as written yields no usable
   candidate, it is searched again with its first part split into name and
   qualifiers ("casentino arezzo, italia" -> "casentino, arezzo, italia"; at
   most two splits: before the last word, then before the last two). A
   descriptor ("vallata del Casentino") is searched without it ("Casentino").
   Every such retry is reported (``matched_query`` / ``fallback_from``).

Lookups are bounded: at most MAX_LOOKUPS_PER_QUERY (4) searches per place
(the text, its bare name, two splits), each for up to LOOKUP_LIMIT (10)
candidates. Nominatim's usage policy is respected: the app's user agent, at
most one request per second process-wide (MIN_REQUEST_INTERVAL_SEC, 1.1 s),
results cached per process (transient errors are not cached).
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from eo_brain.agent.validators import ValidationError, clamp_geocoder_bbox, validate_bbox

logger = logging.getLogger(__name__)

_GEOCODER = Nominatim(
    user_agent="eo-brain/1.0",
    timeout=10,
)

LOOKUP_LIMIT = 10
MAX_LOOKUPS_PER_QUERY = 4
MIN_REQUEST_INTERVAL_SEC = 1.1
AMBIGUITY_RATIO = 0.9
_MAX_LISTED = 5

# Padding in degrees around a centroid when a candidate has no usable bbox.
_DEFAULT_PADDING_DEG = 0.05
_REGION_PADDING_DEG = 0.15
_REGION_TYPES = {"river", "waterway", "valley", "mountain_range", "state", "county"}

# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class GeocodingResult:
    name: str
    lat: float
    lon: float
    bbox: Tuple[float, float, float, float]  # (west, south, east, north)
    source: str  # "nominatim_bbox" | "synthetic" | "stub" | "session"
    raw_type: Optional[str] = None
    matched_query: Optional[str] = None   # the text that matched, when it is not the query as written
    fallback_from: Optional[str] = None   # the query as written, when only a retry matched
    kind: Optional[str] = None            # "region", "village", "city", "river", ... (what it is)
    area_like: bool = True                # False: a non-area feature, used only as a last resort

    def to_list(self) -> list[float]:
        return list(self.bbox)


@dataclass(frozen=True)
class PlaceCandidate:
    """One place the user can choose; ``label`` is the text to pass back as the location."""

    label: str
    kind: str
    where: str
    result: GeocodingResult

    def describe(self) -> str:
        return f"{self.label} ({self.kind})"


class PlaceNotFoundError(ValueError):
    """Nothing usable as an area matched the name (model-fixable: try a corrected name)."""


class PlaceChoiceNeeded(ValueError):
    """
    The text does not decide between places, or the only matches are of
    another kind than the one asked for. Only the user can choose.
    ``reason``: "ambiguous" | "kind".
    """

    def __init__(self, query: str, candidates: Sequence[PlaceCandidate], reason: str, wanted: Optional[str] = None):
        self.query = query
        self.candidates = list(candidates)
        self.reason = reason
        self.wanted = wanted
        listed = "; ".join(f"{i}) {c.describe()}" for i, c in enumerate(self.candidates, 1))
        if reason == "kind":
            text = (f"No {wanted} named as in '{query}' was found; the only matches are of another kind: "
                    f"{listed}. Is one of them meant?")
        else:
            text = (f"'{query}' matches more than one place and nothing in the request says which: "
                    f"{listed}. Which one is meant?")
        super().__init__(text)


class _TransientGeocoderError(Exception):
    """Raised inside the cached function so lru_cache does not store it."""


# ---------------------------------------------------------------- text

def normalize(text: Any) -> str:
    """casefold, strip accents, punctuation (apostrophes too) -> spaces, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(text)).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Generic descriptors around a place name, Italian and English. Only the
# leading descriptor of the first comma-separated part is removed ("vallata del
# Casentino, Arezzo" -> "Casentino, Arezzo"). "Val" is deliberately NOT a
# descriptor: in Italian it is part of proper names (Val d'Orcia, Val di Non,
# Val Gardena), and "Non" or "Orcia" alone is a different place.
_ARTICLE = re.compile(
    r"^(?:(?:the|in|la|il|lo|le|i|gli|nella|nello|nel|nei|negli|nelle|sulla|sul)\s+|l['’]\s*)",
    re.IGNORECASE,
)
_REGION_NOUNS_IT = r"vallata|valle|regione|territorio|comprensorio|piana|pianura"
_PROVINCE_NOUNS_IT = r"provincia"
_PLAIN_NOUNS_IT = r"zona|area|dintorni"
_IT_NOUN = rf"(?:{_REGION_NOUNS_IT}|{_PROVINCE_NOUNS_IT}|{_PLAIN_NOUNS_IT})"
_DESCRIPTORS = [
    # Italian: "<noun> di/del/della/dello/dei/degli/delle/dell'/d' X"
    re.compile(rf"^({_IT_NOUN})\s+(?:di|del|della|dello|dei|degli|delle|de)\s+", re.IGNORECASE),
    re.compile(rf"^({_IT_NOUN})\s+(?:dell|d)['’]\s*", re.IGNORECASE),
    # Italian: "nei dintorni di X", "intorno a X", "vicino a X"
    re.compile(r"^(intorno|attorno|vicino)\s+(?:a|al|alla|allo|ai|agli|alle)\s+", re.IGNORECASE),
    re.compile(r"^(intorno|attorno|vicino)\s+all['’]\s*", re.IGNORECASE),
    # English: "valley of X", "area of X", "surroundings of X", ...; "around X", "near X"
    re.compile(r"^(valley|area|zone|region|province|surroundings|outskirts|vicinity)\s+of\s+(?:the\s+)?",
               re.IGNORECASE),
    re.compile(r"^(?:area|zone|region)\s+(around|near)\s+", re.IGNORECASE),   # surroundings: no kind
    re.compile(r"^(around|near|close\s+to)\s+", re.IGNORECASE),
]
# English trailing descriptors: "Arno valley", "Florence area", "Siena surroundings"
_TRAILING = re.compile(r"\s+(valley|area|zone|region|province|surroundings|outskirts)$", re.IGNORECASE)

_WANT_REGION = {"vallata", "valle", "regione", "territorio", "comprensorio", "piana", "pianura",
                "valley", "region"}
_WANT_PROVINCE = {"provincia", "province"}


def _strip_descriptor(query: str) -> Tuple[Optional[str], Optional[str]]:
    """(``query`` without its generic descriptor, the kind it asks for) or (None, None)."""
    head, sep, tail = query.strip().partition(",")
    name, found, want = head.strip(), False, None
    for _ in range(3):  # "nella zona dei dintorni di X": a few nested descriptors at most
        before = name
        stripped = _ARTICLE.sub("", name).strip()
        for pattern in _DESCRIPTORS + [_TRAILING]:
            m = pattern.search(stripped)
            if m:
                word = m.group(1).casefold()
                if want is None and word in _WANT_REGION:
                    want = "region"
                elif want is None and word in _WANT_PROVINCE:
                    want = "province"
                stripped, found = (stripped[:m.start()] + stripped[m.end():]).strip(), True
        if found:
            name = stripped   # an article alone ("La Spezia") is not a descriptor
        if name == before:
            break
    if not found or not name:
        return None, None
    return (f"{name}{sep}{tail}" if sep else name), want


def bare_place_name(query: str) -> Optional[str]:
    """
    ``query`` without its generic descriptor, or None when it has none.

    "vallata del Casentino" -> "Casentino"; "the area around Siena" -> "Siena";
    "Arno valley, Tuscany" -> "Arno, Tuscany"; "Firenze" -> None.
    """
    return _strip_descriptor(query)[0]


def _first_part(text: str) -> str:
    return _ARTICLE.sub("", text.partition(",")[0].strip()).strip()


# ---------------------------------------------------------------- candidates

_AREA_CLASSES = {"boundary", "place", "natural", "waterway", "water", "landuse", "leisure"}
_POINT_TYPES = {
    ("place", "house"), ("place", "plot"),
    ("natural", "tree"), ("natural", "peak"), ("natural", "spring"), ("natural", "rock"),
    ("natural", "stone"), ("natural", "cave_entrance"), ("natural", "saddle"), ("natural", "tree_row"),
    ("waterway", "waterfall"), ("waterway", "dam"), ("waterway", "weir"), ("waterway", "lock_gate"),
}
_SETTLEMENTS = {"city", "town", "village", "hamlet", "suburb", "quarter", "neighbourhood", "borough",
                "city_district", "isolated_dwelling", "croft", "farm", "allotments", "locality"}
_PROVINCES = {"county", "province", "state_district", "state"}
_KIND_LABELS = {"county": "province or county", "state": "state or region", "state_district": "district",
                "secondary": "road", "primary": "road", "tertiary": "road", "residential": "road",
                "road": "road", "tourism": "point of interest", "natural": "natural feature"}
# namedetails keys that are names; short_name and ref hold codes ("PO", "FI").
_NAME_KEY = re.compile(r"^(name|name:[\w-]+|alt_name(:[\w-]+)?|official_name(:[\w-]+)?|int_name|"
                       r"loc_name|old_name(:[\w-]+)?|reg_name|nat_name)$")

EXACT, PARTIAL, NONE = 2, 1, 0


@dataclass
class _Candidate:
    raw: Dict[str, Any]
    order: int
    cls: str = ""
    typ: str = ""
    addresstype: str = ""
    importance: float = 0.0
    names: List[str] = field(default_factory=list)
    label: str = ""
    lat: float = 0.0
    lon: float = 0.0
    box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # as Nominatim gives it (W, S, E, N)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # clamped, for the analysis
    source: str = "nominatim_bbox"
    usable: bool = False

    @property
    def kind(self) -> str:
        key = self.addresstype or self.typ or self.cls
        return _KIND_LABELS.get(key, key.replace("_", " "))

    @property
    def area_like(self) -> bool:
        return self.cls in _AREA_CLASSES and (self.cls, self.typ) not in _POINT_TYPES

    @property
    def settlement(self) -> bool:
        return (self.addresstype or self.typ) in _SETTLEMENTS

    @property
    def province(self) -> bool:
        return self.cls in ("boundary", "place") and (self.addresstype or self.typ) in _PROVINCES

    def fits(self, want: Optional[str]) -> bool:
        if want == "region":
            return not self.settlement
        if want == "province":
            return self.province
        return True

    def match(self, name: str) -> int:
        wanted = normalize(name)
        if not wanted:
            return NONE
        tokens = set(wanted.split())
        best = NONE
        for n in self.names:
            if n == wanted:
                return EXACT
            nt = set(n.split())
            if nt and (tokens <= nt or nt <= tokens):
                best = PARTIAL
        return best

    def where(self) -> str:
        addr = self.raw.get("address") or {}
        own = normalize(self.raw.get("name") or "")
        parts = []
        for key in ("county", "state", "country"):
            v = addr.get(key)
            if v and normalize(v) != own and v not in parts:
                parts.append(v)
        if parts:
            return ", ".join(parts)
        return ", ".join(p.strip() for p in self.label.split(",")[1:]) or self.label

    def result(self, **extra) -> GeocodingResult:
        return GeocodingResult(
            name=self.label, lat=self.lat, lon=self.lon, bbox=self.bbox, source=self.source,
            raw_type=self.typ or None, kind=self.kind, area_like=self.area_like, **extra,
        )

    def as_choice(self) -> PlaceCandidate:
        return PlaceCandidate(label=self.label, kind=self.kind, where=self.where(), result=self.result())


def _names(raw: Dict[str, Any]) -> List[str]:
    values = [raw.get("name") or ""]
    for key, value in (raw.get("namedetails") or {}).items():
        if _NAME_KEY.match(key) and isinstance(value, str):
            values.extend(value.split(";"))
    if not any(values):
        values.append((raw.get("display_name") or "").split(",")[0])
    out: List[str] = []
    for v in values:
        n = normalize(v)
        if n and n not in out:
            out.append(n)
    return out


def _usable_extent(bbox: Tuple[float, float, float, float]) -> bool:
    try:
        validate_bbox(list(bbox), warn=False)
        return True
    except ValidationError:
        return False


def _candidate(raw: Dict[str, Any], order: int) -> Optional[_Candidate]:
    try:
        lat, lon = float(raw["lat"]), float(raw["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    c = _Candidate(
        raw=raw, order=order,
        cls=str(raw.get("category") or raw.get("class") or ""),
        typ=str(raw.get("type") or ""),
        addresstype=str(raw.get("addresstype") or ""),
        importance=float(raw.get("importance") or 0.0),
        names=_names(raw),
        label=str(raw.get("display_name") or raw.get("name") or ""),
        lat=lat, lon=lon,
    )
    box = None
    raw_bbox = raw.get("boundingbox")
    if raw_bbox and len(raw_bbox) == 4:
        try:
            s, n, w, e = (float(v) for v in raw_bbox)
            box = (max(w, -180.0), max(s, -90.0), min(e, 180.0), min(n, 90.0))
        except (TypeError, ValueError):
            box = None
    if box is None or box[0] >= box[2] or box[1] >= box[3]:
        pad = _REGION_PADDING_DEG if c.typ in _REGION_TYPES else _DEFAULT_PADDING_DEG
        box = (max(lon - pad, -180.0), max(lat - pad, -90.0), min(lon + pad, 180.0), min(lat + pad, 90.0))
        c.source = "synthetic"
    c.box = box
    # Clamp oversized boxes (Nominatim returns province-level boxes for some cities).
    clamped = clamp_geocoder_bbox(box)
    c.bbox = (clamped[0], clamped[1], clamped[2], clamped[3])
    c.usable = _usable_extent(c.bbox)
    return c


def _same_place(a: _Candidate, b: _Candidate) -> bool:
    def inside(p: _Candidate, q: _Candidate) -> bool:
        w, s, e, n = q.box
        return w <= p.lon <= e and s <= p.lat <= n
    return inside(a, b) or inside(b, a)


# ---------------------------------------------------------------- resolution

@dataclass(frozen=True)
class _Attempt:
    text: str
    name: str                 # the name to match: first part, no article or descriptor
    want: Optional[str]       # "region" | "province" | None
    retried: bool             # not the text as written


def _attempts(query: str) -> List[_Attempt]:
    """The searches for one place, in order: as written, without descriptor, name/qualifier splits."""
    text = " ".join(query.split())
    bare, want = _strip_descriptor(text)
    base = bare or text
    attempts = [_Attempt(text, _first_part(base), want, False)]
    if bare:
        attempts.append(_Attempt(bare, _first_part(bare), want, True))
    head, sep, tail = base.partition(",")
    words = head.split()
    for k in (len(words) - 1, len(words) - 2):
        if k < 1:
            continue
        parts = [" ".join(words[:k]), " ".join(words[k:])] + ([tail.strip()] if tail.strip() else [])
        attempts.append(_Attempt(", ".join(parts), _first_part(" ".join(words[:k])), want, True))
    seen, unique = set(), []
    for a in attempts:
        key = " ".join(a.text.split()).casefold()   # punctuation matters to the search
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique[:MAX_LOOKUPS_PER_QUERY]


def _decide(query: str, attempt: _Attempt, cands: List[_Candidate]) -> Optional[_Candidate]:
    """The chosen candidate, None when nothing usable matched, or raise PlaceChoiceNeeded."""
    named = [(c, c.match(attempt.name)) for c in cands if c.area_like and c.usable]
    named = [(c, m) for c, m in named if m != NONE]
    if not named:
        return None
    if attempt.want:
        fitting = [(c, m) for c, m in named if c.fits(attempt.want)]
        if not fitting:
            listed = _distinct(sorted((c for c, _ in named), key=lambda c: (-c.importance, c.order)))
            raise PlaceChoiceNeeded(query, [c.as_choice() for c in listed], "kind", wanted=attempt.want)
        named = fitting
    exact = [c for c, m in named if m == EXACT]
    pool = exact or [c for c, _ in named]
    top = max(pool, key=lambda c: (c.importance, -c.order))
    rivals = [c for c in exact if c is not top and not _same_place(c, top)
              and c.importance >= AMBIGUITY_RATIO * top.importance]
    if rivals:
        listed = _distinct([top] + sorted(rivals, key=lambda c: (-c.importance, c.order)))
        raise PlaceChoiceNeeded(query, [c.as_choice() for c in listed], "ambiguous")
    return top


def _distinct(cands: Sequence[_Candidate]) -> List[_Candidate]:
    out: List[_Candidate] = []
    for c in cands:
        if not any(_same_place(c, o) for o in out):
            out.append(c)
    return out[:_MAX_LISTED]


class PlaceResolver:
    """
    Resolve a place name with any ``search(text) -> list of Nominatim-like
    dicts``. Production uses the throttled, cached Nominatim search; the evals
    and tests use recorded responses or a gazetteer.
    """

    def __init__(self, search: Callable[[str], Sequence[Dict[str, Any]]]):
        self.search = search

    def resolve(self, query: str) -> GeocodingResult:
        """The chosen area, or raise PlaceNotFoundError / PlaceChoiceNeeded."""
        rejected: List[_Candidate] = []       # found, but not an area, or no name match
        fallback: List[Tuple[_Attempt, _Candidate]] = []
        tried: List[str] = []
        for attempt in _attempts(query):
            tried.append(attempt.text)
            try:
                raws = self.search(attempt.text)
            except _TransientGeocoderError as exc:
                raise PlaceNotFoundError(
                    f"Could not geocode '{query}': the geocoding service did not answer ({exc}). Try again."
                ) from exc
            cands = [c for c in (_candidate(r, i) for i, r in enumerate(raws)) if c is not None]
            chosen = _decide(query, attempt, cands)
            if chosen is not None:
                return self._result(query, attempt, chosen)
            for c in cands:
                rejected.append(c)
                if (not c.area_like and c.usable and c.source == "nominatim_bbox"
                        and c.match(attempt.name) != NONE and c.fits(attempt.want)):
                    fallback.append((attempt, c))
        if fallback:
            # Last resort: no area matched, but a named feature whose box can be analysed.
            attempt, c = max(fallback, key=lambda ac: (ac[1].importance, -ac[1].order))
            return self._result(query, attempt, c)
        raise PlaceNotFoundError(self._not_found_text(query, tried, rejected))

    @staticmethod
    def _result(query: str, attempt: _Attempt, c: _Candidate) -> GeocodingResult:
        if attempt.retried:
            logger.info(f"Geocoded '{query}' via '{attempt.text}' -> {c.label} ({c.kind})")
            return c.result(matched_query=attempt.text, fallback_from=query)
        return c.result()

    @staticmethod
    def _not_found_text(query: str, tried: List[str], rejected: List[_Candidate]) -> str:
        others = [t for t in tried[1:]]
        also = f" (also tried {', '.join(repr(t) for t in others)})" if others else ""
        text = f"Could not geocode '{query}'{also}."
        seen: List[str] = []
        for c in sorted(rejected, key=lambda c: (-c.importance, c.order)):
            d = f"'{c.label.split(',')[0]}' ({c.kind})"
            if d not in seen:
                seen.append(d)
        if seen:
            text += (f" The search found only {', '.join(seen[:3])}: none of them is an area with "
                     f"that name. ")
        else:
            text += " "
        return text + "Try a more specific place name (e.g. 'Rome, Italy' instead of 'Roma')."


# ---------------------------------------------------------------- Nominatim

_REQUEST_LOCK = threading.Lock()
_last_request_at = [float("-inf")]
_sleep = time.sleep
_monotonic = time.monotonic


def _throttle() -> None:
    """Nominatim policy: at most one request per second, across the whole process."""
    with _REQUEST_LOCK:
        wait = _last_request_at[0] + MIN_REQUEST_INTERVAL_SEC - _monotonic()
        if wait > 0:
            _sleep(wait)
        _last_request_at[0] = _monotonic()


@lru_cache(maxsize=512)
def _search_cached(key: str) -> Tuple[Dict[str, Any], ...]:
    _throttle()
    try:
        locations = _GEOCODER.geocode(
            key,
            exactly_one=False,
            limit=LOOKUP_LIMIT,
            language="en",
            addressdetails=True,
            namedetails=True,
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning(f"Geocoding failed for '{key}': {exc}")
        raise _TransientGeocoderError(str(exc)) from exc
    rows = tuple((loc.raw or {}) for loc in (locations or []))
    logger.info(f"Nominatim '{key}': {len(rows)} candidate(s)")
    return rows


def nominatim_search(text: str) -> Tuple[Dict[str, Any], ...]:
    """Throttled, cached Nominatim search (the cache key ignores case and spacing)."""
    return _search_cached(" ".join(text.split()).casefold())


_RESOLVER = PlaceResolver(nominatim_search)


def geocode_or_fail(query: str) -> GeocodingResult:
    """
    Resolve a place name to an area. Raises PlaceNotFoundError (a ValueError,
    model-fixable) or PlaceChoiceNeeded (a ValueError the agent turns into a
    question for the user). Use in tool context.
    """
    return _RESOLVER.resolve(query)


def geocode(query: str) -> Optional[GeocodingResult]:
    """Like geocode_or_fail, but None when nothing is chosen (not found, or a choice is needed)."""
    try:
        return geocode_or_fail(query)
    except ValueError:
        return None
