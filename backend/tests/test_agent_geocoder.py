"""
Unit tests for eo_brain.agent.geocoder (no network).

Most cases replay real Nominatim responses recorded on 2026-11-27
(evals/fixtures/nominatim_recorded.json), including the two phrasings that
launched analyses in the wrong place in live use. The rest use small
synthetic responses for rules the recordings do not exercise.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from geopy.exc import GeocoderServiceError, GeocoderTimedOut

from eo_brain.agent import geocoder
from eo_brain.agent.geocoder import (
    MAX_LOOKUPS_PER_QUERY, PlaceChoiceNeeded, PlaceNotFoundError, PlaceResolver,
)
from eo_brain.agent.validators import estimate_area_km2

FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "nominatim_recorded.json"
RECORDED = json.loads(FIXTURE.read_text(encoding="utf-8"))["responses"]

REGION = "Casentino, Arezzo, Tuscany, Italy"
VILLAGE = "Casentino, Sant'Eusanio Forconese, L'Aquila, Abruzzo, 67040, Italy"


def _key(text):
    return " ".join(text.split()).casefold()


class Recorded:
    """search() over the recorded responses, keyed as the production cache keys them."""

    def __init__(self, extra=None):
        self.responses = {_key(k): v for k, v in RECORDED.items()}
        self.responses.update({_key(k): v for k, v in (extra or {}).items()})
        self.searches = []

    def search(self, text):
        self.searches.append(text)
        return self.responses.get(_key(text), [])


def resolve(query, extra=None):
    rec = Recorded(extra)
    try:
        return PlaceResolver(rec.search).resolve(query), rec.searches
    except ValueError as e:
        e.searches = rec.searches
        raise


# ---------------------------------------------------------------- the live failures

def test_valley_phrasing_reaches_the_region_not_the_village():
    # Live: the bare-name retry took Nominatim's first result, a village in Abruzzo.
    assert RECORDED["Casentino"][0]["display_name"] == VILLAGE
    res, searches = resolve("vallata del Casentino")
    assert res.name == REGION and res.kind == "region"
    assert searches == ["vallata del Casentino", "Casentino"]
    assert res.fallback_from == "vallata del Casentino" and res.matched_query == "Casentino"


def test_comma_less_phrasing_reaches_the_region_not_a_street():
    # Live: the first result for the text as written is a street, Viale Italia.
    assert RECORDED["casentino arezzo, italia"][0]["class"] == "highway"
    res, searches = resolve("casentino arezzo, italia")
    assert res.name == REGION and res.kind == "region"
    assert searches == ["casentino arezzo, italia", "casentino, arezzo, italia"]
    assert res.fallback_from == "casentino arezzo, italia"
    assert res.matched_query == "casentino, arezzo, italia"


def test_punctuation_does_not_change_the_place():
    with_commas, searches = resolve("Casentino, Arezzo, Italia")
    without, _ = resolve("casentino arezzo, italia")
    assert searches == ["Casentino, Arezzo, Italia"]                 # one lookup
    assert with_commas.fallback_from is None and with_commas.matched_query is None
    assert with_commas.name == without.name == REGION
    assert with_commas.bbox == without.bbox


def test_bare_ambiguous_name_asks_with_kind_and_province():
    with pytest.raises(PlaceChoiceNeeded) as exc:
        resolve("Casentino")
    e = exc.value
    assert e.reason == "ambiguous" and e.searches == ["Casentino"]
    assert [(c.label, c.kind) for c in e.candidates] == [(VILLAGE, "village"), (REGION, "region")]
    assert [c.where for c in e.candidates] == ["L'Aquila, Abruzzo, Italy", "Arezzo, Tuscany, Italy"]
    assert "1) " + VILLAGE + " (village)" in str(e) and "2) " + REGION + " (region)" in str(e)


@pytest.mark.parametrize("label", [VILLAGE, REGION])
def test_each_listed_label_resolves_to_exactly_that_candidate(label):
    with pytest.raises(PlaceChoiceNeeded) as exc:
        resolve("Casentino")
    listed = {c.label: c for c in exc.value.candidates}
    res, searches = resolve(label)
    assert searches == [label]
    assert res.name == label and res.bbox == listed[label].result.bbox


# ---------------------------------------------------------------- cases that already worked

@pytest.mark.parametrize("query", ["Firenze", "Florence, Italy"])
def test_a_city_with_or_without_country(query):
    res, searches = resolve(query)
    assert len(searches) == 1 and res.kind == "city" and res.area_like
    assert res.name == "Florence, Tuscany, Italy"
    assert res.bbox == pytest.approx((11.1498, 43.7262, 11.3398, 43.8356), abs=1e-3)
    assert res.to_list() == list(res.bbox)


def test_nested_places_of_one_name_are_not_a_choice():
    # Florence the city and the metropolitan city: one place, the city wins.
    res, _ = resolve("Firenze")
    assert res.kind == "city"


@pytest.mark.parametrize("query,name,kind", [
    ("Lago di Garda", "Lake Garda, Brescia, Lombardy, Italy", "lake"),
    ("Arno", "Arno, Tuscany, Italy", "river"),
])
def test_a_lake_or_river(query, name, kind):
    res, searches = resolve(query)
    assert (res.name, res.kind) == (name, kind) and len(searches) == 1


def test_a_far_less_important_homonym_is_not_a_choice():
    # Valencia (Spain) 0.73 vs Valencia (Venezuela) 0.55: below the 90 % ratio.
    res, _ = resolve("Valencia")
    assert res.name.endswith("Spain") and res.kind == "city"


def test_province_descriptor_takes_the_province():
    res, _ = resolve("provincia di Arezzo")
    assert res.name == "Arezzo, Tuscany, Italy" and res.kind == "province or county"


def test_misspelling_is_not_found_and_says_what_was_tried():
    with pytest.raises(PlaceNotFoundError, match=r"Could not geocode 'Firenez'\. Try a more specific") as exc:
        resolve("Firenez")
    assert exc.value.searches == ["Firenez"]


def test_other_names_never_win_on_codes_or_other_words():
    # "Po": Nominatim offers Prato (short_name "PO"), Pau, ... but no area named Po.
    with pytest.raises(PlaceNotFoundError) as exc:
        resolve("Po")
    assert "'Prato' (province or county)" in str(exc.value)
    assert "none of them is an area with that name" in str(exc.value)


# ---------------------------------------------------------------- rules, on synthetic responses

def _raw(name, cls="boundary", typ="administrative", addresstype=None, importance=0.5,
         lat=43.7, lon=11.8, box=0.1, display=None, **names):
    return {
        "class": cls, "type": typ, "addresstype": addresstype or typ, "importance": importance,
        "name": name, "display_name": display or f"{name}, Somewhere",
        "namedetails": {"name": name, **names},
        "lat": str(lat), "lon": str(lon),
        "boundingbox": [str(lat - box), str(lat + box), str(lon - box), str(lon + box)],
    }


def resolve_with(responses, query):
    calls = []

    def search(text):
        calls.append(text)
        return responses.get(text, [])

    return PlaceResolver(search).resolve(query), calls


def test_an_area_beats_a_more_important_road():
    road = _raw("Casentino", cls="highway", typ="secondary", importance=0.9)
    place = _raw("Casentino", cls="place", typ="region", importance=0.3)
    res, _ = resolve_with({"Casentino": [road, place]}, "Casentino")
    assert res.kind == "region" and res.area_like


def test_a_non_area_is_a_last_resort_and_says_what_it_is():
    road = _raw("Via Roma", cls="highway", typ="residential", addresstype="road", box=0.01)
    res, _ = resolve_with({"Via Roma": [road]}, "Via Roma")
    assert res.area_like is False and res.kind == "road"


def _no_box(raw):
    del raw["boundingbox"]
    return raw


@pytest.mark.parametrize("raw", [
    _raw("Quercia", cls="natural", typ="tree", box=0.00001),             # a single tree
    _raw("Via Stretta", cls="highway", typ="residential", box=0.0001),   # too narrow to analyse
    _no_box(_raw("Museo", cls="tourism", typ="museum")),                 # a point: padding is not an extent
])
def test_points_and_slivers_are_never_used(raw):
    with pytest.raises(PlaceNotFoundError):
        resolve_with({raw["name"]: [raw]}, raw["name"])


def test_region_descriptor_never_silently_becomes_a_village():
    village = _raw("Pratomagno", cls="place", typ="village")
    with pytest.raises(PlaceChoiceNeeded) as exc:
        resolve_with({"valle del Pratomagno": [], "Pratomagno": [village]}, "valle del Pratomagno")
    assert exc.value.reason == "kind" and exc.value.wanted == "region"
    assert [c.kind for c in exc.value.candidates] == ["village"]
    assert "No region named as in 'valle del Pratomagno'" in str(exc.value)


def test_plain_descriptor_states_no_kind():
    village = _raw("Pratomagno", cls="place", typ="village")
    res, calls = resolve_with({"Pratomagno": [village]}, "zona del Pratomagno")
    assert res.kind == "village" and calls == ["zona del Pratomagno", "Pratomagno"]
    assert res.fallback_from == "zona del Pratomagno"


@pytest.mark.parametrize("ratio,asks", [(0.89, False), (0.90, True)])
def test_ambiguity_threshold(ratio, asks):
    a = _raw("Montalto", importance=0.5, lat=43.0, lon=11.0)
    b = _raw("Montalto", importance=0.5 * ratio, lat=38.0, lon=16.0, display="Montalto, Elsewhere")
    if asks:
        with pytest.raises(PlaceChoiceNeeded):
            resolve_with({"Montalto": [a, b]}, "Montalto")
    else:
        assert resolve_with({"Montalto": [a, b]}, "Montalto")[0].lat == 43.0


def test_exact_name_beats_a_partial_one():
    partial = _raw("Lago di Bilancino", cls="water", typ="lake", importance=0.6)
    exact = _raw("Bilancino", cls="water", typ="reservoir", importance=0.2, lat=44.0)
    res, _ = resolve_with({"Bilancino": [partial, exact]}, "Bilancino")
    assert res.lat == 44.0


def test_short_name_and_ref_are_not_names():
    prato = _raw("Prato", short_name="PO", ref="PO")
    with pytest.raises(PlaceNotFoundError):
        resolve_with({"Po": [prato]}, "Po")


def test_found_as_written_is_not_retried():
    _, calls = resolve_with({"Monte Amiata": [_raw("Monte Amiata")]}, "Monte Amiata")
    assert calls == ["Monte Amiata"]


@pytest.mark.parametrize("query", [
    "vallata del alto casentino arezzo, italia",
    "nella vallata del fiume alto casentino arezzo pratovecchio stia, italia",
])
def test_lookups_per_place_are_bounded(query):
    calls = []
    with pytest.raises(PlaceNotFoundError):
        PlaceResolver(lambda t: calls.append(t) or []).resolve(query)
    assert len(calls) == MAX_LOOKUPS_PER_QUERY == 4


def test_retry_order_bare_name_then_splits():
    calls = []
    with pytest.raises(PlaceNotFoundError):
        PlaceResolver(lambda t: calls.append(t) or []).resolve("vallata del alto casentino arezzo, italia")
    assert calls == ["vallata del alto casentino arezzo, italia", "alto casentino arezzo, italia",
                     "alto casentino, arezzo, italia", "alto, casentino arezzo, italia"]


def test_a_one_word_name_is_searched_once():
    calls = []
    with pytest.raises(PlaceNotFoundError):
        PlaceResolver(lambda t: calls.append(t) or []).resolve("Atlantide")
    assert calls == ["Atlantide"]


def test_all_failed_attempts_are_named_in_the_error():
    with pytest.raises(PlaceNotFoundError, match=r"Could not geocode 'zona di Xyzzy' \(also tried 'Xyzzy'\)"):
        resolve_with({}, "zona di Xyzzy")


def test_oversized_box_is_clamped():
    big = _raw("Grande", box=5.0)
    res, _ = resolve_with({"Grande": [big]}, "Grande")
    assert estimate_area_km2(list(res.bbox)) <= 50000


def test_missing_box_gets_a_padded_one():
    raw = _raw("Senzabox")
    del raw["boundingbox"]
    res, _ = resolve_with({"Senzabox": [raw]}, "Senzabox")
    assert res.source == "synthetic"
    assert res.bbox == pytest.approx((11.75, 43.65, 11.85, 43.75))


# ---------------------------------------------------------------- descriptors

@pytest.mark.parametrize("query,bare", [
    ("vallata del Casentino", "Casentino"),
    ("Vallata del Casentino, Arezzo", "Casentino, Arezzo"),
    ("la valle dell'Arno", "Arno"),
    ("zona di Prato", "Prato"),
    ("area di Pisa", "Pisa"),
    ("provincia di Arezzo", "Arezzo"),
    ("dintorni di Siena", "Siena"),
    ("nei dintorni di Siena", "Siena"),
    ("territorio del Chianti", "Chianti"),
    ("intorno a Firenze", "Firenze"),
    ("valley of the Arno", "Arno"),
    ("area of Florence", "Florence"),
    ("zone of Pisa", "Pisa"),
    ("province of Arezzo", "Arezzo"),
    ("surroundings of Lucca", "Lucca"),
    ("the area around Siena", "Siena"),
    ("near Pisa", "Pisa"),
    ("Arno valley, Tuscany", "Arno, Tuscany"),
    ("Florence area", "Florence"),
])
def test_bare_place_name_strips_generic_descriptors(query, bare):
    assert geocoder.bare_place_name(query) == bare


@pytest.mark.parametrize("query", [
    "Firenze", "Casentino", "La Spezia", "Le Marche",
    "Val d'Orcia", "Val di Non",            # "Val" is part of Italian proper names
    "area", "valley",                       # nothing left after the descriptor
])
def test_bare_place_name_leaves_plain_names_alone(query):
    assert geocoder.bare_place_name(query) is None


@pytest.mark.parametrize("query,want", [
    ("vallata del Casentino", "region"), ("valle dell'Arno", "region"), ("Arno valley", "region"),
    ("regione Toscana", None),               # no "di": not a descriptor, the name as written
    ("piana di Lucca", "region"), ("territorio del Chianti", "region"),
    ("provincia di Arezzo", "province"), ("province of Siena", "province"),
    ("zona di Prato", None), ("dintorni di Siena", None), ("the area around Siena", None),
    ("Firenze", None),
])
def test_descriptor_states_the_kind(query, want):
    assert geocoder._strip_descriptor(query)[1] == want


# ---------------------------------------------------------------- Nominatim plumbing

class FakeNominatim:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.queries = []

    def geocode(self, query, **kwargs):
        self.queries.append((query, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return [SimpleNamespace(raw=r) for r in item]


@pytest.fixture
def nominatim(monkeypatch):
    geocoder._search_cached.cache_clear()
    clock = SimpleNamespace(now=1000.0, sleeps=[])

    def sleep(sec):
        clock.sleeps.append(round(sec, 3))
        clock.now += sec

    monkeypatch.setattr(geocoder, "_monotonic", lambda: clock.now)
    monkeypatch.setattr(geocoder, "_sleep", sleep)
    monkeypatch.setattr(geocoder, "_last_request_at", [float("-inf")])

    def install(*responses):
        fake = FakeNominatim(*responses)
        monkeypatch.setattr(geocoder, "_GEOCODER", fake)
        fake.clock = clock
        return fake

    yield install
    geocoder._search_cached.cache_clear()


def test_request_parameters_and_user_agent(nominatim):
    real_agent = geocoder._GEOCODER.headers["User-Agent"]
    fake = nominatim(RECORDED["Firenze"])
    geocoder.geocode_or_fail("Firenze")
    assert fake.queries == [("firenze", {"exactly_one": False, "limit": 10, "language": "en",
                                         "addressdetails": True, "namedetails": True})]
    assert real_agent == "eo-brain/1.0"


def test_results_are_cached_ignoring_case_and_spacing(nominatim):
    fake = nominatim(RECORDED["Firenze"])
    a = geocoder.geocode_or_fail("Firenze")
    b = geocoder.geocode_or_fail("  firenze ")
    assert a == b and len(fake.queries) == 1


def test_requests_are_at_least_1_1_seconds_apart(nominatim):
    fake = nominatim(RECORDED["vallata del Casentino"], RECORDED["Casentino"])
    assert geocoder.geocode_or_fail("vallata del Casentino").name == REGION
    assert [q for q, _ in fake.queries] == ["vallata del casentino", "casentino"]
    assert fake.clock.sleeps == [1.1]                        # the second request waited


@pytest.mark.parametrize("exc", [GeocoderTimedOut("slow"), GeocoderServiceError("503")])
def test_transient_errors_are_reported_and_not_cached(nominatim, exc):
    fake = nominatim(exc, RECORDED["Firenze"])
    with pytest.raises(PlaceNotFoundError, match="did not answer"):
        geocoder.geocode_or_fail("Firenze")
    assert geocoder.geocode_or_fail("Firenze").kind == "city"   # retried, not served from cache
    assert len(fake.queries) == 2


def test_geocode_returns_none_when_nothing_is_chosen(nominatim):
    nominatim(RECORDED["Casentino"], [])
    assert geocoder.geocode("Casentino") is None      # a choice is needed
    assert geocoder.geocode("Atlantide") is None      # not found
