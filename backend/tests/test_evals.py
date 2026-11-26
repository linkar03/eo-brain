"""
Tests for the eval harness itself: the stub geocoder, the scorer, and the
runner. The scorer must catch wrong output, otherwise a 100% scripted run
means nothing.
"""

import json

import pytest
import yaml

from eo_brain.agent.brain import AgentResult, ToolInvocation
from evals import run as runner
from evals.scoring import check_constraint, score_case, summarize
from evals.stubs import StubGeocoder, contains_phrase, normalize

GAZ = [
    {"name": "Firenze, Toscana, Italia", "aliases": ["firenze", "florence"], "bbox": [11.15, 43.72, 11.33, 43.83]},
    {"name": "Lago di Garda, Italia", "aliases": ["lago di garda", "garda"], "bbox": [10.55, 45.43, 10.88, 45.88]},
]


class TestStubs:
    def test_normalize(self):
        assert normalize("  Città di FIRENZE, Italia! ") == "citta di firenze italia"

    def test_contains_phrase_is_whole_word(self):
        assert contains_phrase("Firenze, Italia", "firenze")
        assert not contains_phrase("Firenzeeee", "firenze")

    def test_geocoder_matches_aliases_in_longer_queries(self):
        geo = StubGeocoder(GAZ)
        assert geo("Florence, Italy").name == "Firenze, Toscana, Italia"
        assert geo("il Lago di Garda").name == "Lago di Garda, Italia"
        assert geo("Firenze, Toscana, Italia").name == "Firenze, Toscana, Italia"  # rerun path
        assert geo.queries == ["Florence, Italy", "il Lago di Garda", "Firenze, Toscana, Italia"]

    def test_geocoder_miss_raises_like_the_real_one(self):
        with pytest.raises(ValueError, match="Could not geocode"):
            StubGeocoder(GAZ)("xqzvbnm")


class TestConstraints:
    @pytest.mark.parametrize(
        "value,constraint,ok",
        [
            ("ndvi", {"equals": "ndvi"}, True),
            ("NDVI", {"equals": "ndvi"}, True),
            ("ndwi", {"equals": "ndvi"}, False),
            (30, {"equals": 30}, True),
            ("mndwi", {"any_of": ["ndwi", "mndwi"]}, True),
            ("ndvi", {"any_of": ["ndwi", "mndwi"]}, False),
            (90, {"range": [89, 92]}, True),
            (93, {"range": [89, 92]}, False),
            ("90", {"range": [89, 92]}, False),
            ("Florence, Italy", {"matches": ["florence", "firenze"]}, True),
            ("Fiesole", {"matches": ["florence", "firenze"]}, False),
            (None, {"matches": ["x"]}, False),
        ],
    )
    def test_check_constraint(self, value, constraint, ok):
        assert check_constraint(value, constraint) is ok

    def test_unknown_constraint(self):
        with pytest.raises(ValueError):
            check_constraint(1, {"between": [0, 2]})


def _result(calls, job_id=None):
    return AgentResult(text="t", job_id=job_id, tool_calls=calls)


def _launch(args, job=True):
    result = {"job_id": "j1", "algorithm": "ndvi", "location": "Firenze"} if job else {"error": "x"}
    return ToolInvocation("launch_analysis", args, result)


CASE = {
    "id": "c", "lang": "it", "input": "ndvi firenze ultimi 3 mesi",
    "expect": {
        "type": "tool_call", "tool": "launch_analysis",
        "args": {"algorithm": {"equals": "ndvi"}, "days_back": {"range": [89, 92]}, "location": {"matches": ["firenze"]}},
    },
}


class TestScoring:
    def test_correct_call_passes(self):
        r = score_case(CASE, _result([_launch({"location": "Firenze", "algorithm": "vegetation", "days_back": 90})], "j1"))
        assert r.passed and r.tool_ok and r.args_ok

    def test_wrong_tool_fails(self):
        r = score_case(CASE, _result([ToolInvocation("list_capabilities", {}, {})]))
        assert not r.passed and r.tool_ok is False and r.args_ok is False

    def test_wrong_window_fails_args_only(self):
        r = score_case(CASE, _result([_launch({"location": "Firenze", "algorithm": "ndvi"})], "j1"))
        assert r.tool_ok and not r.args_ok and not r.passed
        assert r.arg_checks == {"algorithm": True, "days_back": False, "location": True}

    def test_missing_job_fails(self):
        r = score_case(CASE, _result([_launch({"location": "Firenze", "algorithm": "ndvi", "days_back": 90}, job=False)]))
        assert r.args_ok and not r.passed

    def test_no_call_fails(self):
        assert not score_case(CASE, _result([])).passed

    def test_refusal(self):
        case = {"id": "r", "expect": {"type": "refusal"}}
        assert score_case(case, _result([])).passed
        assert score_case(case, _result([_launch({}, job=False)])).passed
        bad = score_case(case, _result([_launch({"location": "x", "algorithm": "ndvi"})], "j1"))
        assert not bad.passed and bad.refused_ok is False

    def test_summary(self):
        results = [
            score_case(CASE, _result([_launch({"location": "Firenze", "algorithm": "ndvi", "days_back": 90})], "j1")),
            score_case(CASE, _result([])),
            score_case({"id": "r", "expect": {"type": "refusal"}}, _result([])),
        ]
        s = summarize(results)
        assert s["passed"] == 2
        assert s["tool_selection_accuracy"] == {"correct": 1, "total": 2, "rate": 0.5}
        assert s["correct_refusal_rate"]["rate"] == 1.0
        assert s["false_refusals"] == 1


class TestDataset:
    def test_shipped_dataset_is_well_formed(self):
        data = runner.load_dataset(runner.DEFAULT_CASES)
        langs = {c["lang"] for c in data["cases"]}
        kinds = {c["expect"]["type"] for c in data["cases"]}
        assert langs == {"it", "en"} and kinds == {"tool_call", "refusal"}
        for case in data["cases"]:
            for turn in [*case.get("setup", []), case]:
                assert runner.scripted_steps(turn), case["id"]


class TestRunner:
    def test_scripted_run_writes_report(self, tmp_path, capsys):
        report = tmp_path / "r.json"
        code = runner.main(["--provider", "scripted", "--report", str(report)])
        out = capsys.readouterr().out
        assert code == 0
        assert "NOT a measure of model quality" in out
        data = json.loads(report.read_text())
        assert data["measures_model_quality"] is False
        assert data["summary"]["passed"] == data["summary"]["cases"]

    def test_scripted_run_fails_when_a_case_fails(self, tmp_path, capsys):
        data = runner.load_dataset(runner.DEFAULT_CASES)
        case = next(c for c in data["cases"] if c["id"] == "it-compact-ndvi-firenze")
        case["expect"]["args"]["algorithm"] = {"equals": "flood"}  # deliberately wrong gold
        data["cases"] = [case]
        path = tmp_path / "cases.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        code = runner.main(["--provider", "scripted", "--cases", str(path), "--report", str(tmp_path / "r.json")])
        assert code == 1
        assert "FAIL" in capsys.readouterr().out

    def test_live_mode_requires_key(self, monkeypatch, capsys):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert runner.main(["--provider", "gemini", "--report", "/dev/null"]) == 2
