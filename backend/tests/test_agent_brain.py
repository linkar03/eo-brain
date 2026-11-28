"""
Unit tests for the function-calling loop in eo_brain.agent.brain.

The model is a ScriptedProvider, the geocoder and the job repository are
in-memory stubs: no network, no database, no API key.
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from eo_brain.agent import brain
from eo_brain.agent.brain import Agent, SYSTEM_PROMPT, TOOL_SPECS
from eo_brain.agent.llm import (
    ModelTurn,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolResults,
    UserMessage,
)
from eo_brain.agent.providers.scripted import ScriptedProvider, relay_tool_results
from eo_brain.agent.session import SessionStore
from evals.stubs import FIXED_NOW, InMemoryJobRepository, StubGeocoder

GAZETTEER = [
    {"name": "Firenze, Toscana, Italia", "aliases": ["firenze", "florence"], "bbox": [11.15, 43.72, 11.33, 43.83]},
    {"name": "Pisa, Toscana, Italia", "aliases": ["pisa"], "bbox": [10.33, 43.66, 10.48, 43.76]},
    {"name": "Bologna, Emilia-Romagna, Italia", "aliases": ["bologna"], "bbox": [11.23, 44.42, 11.43, 44.56]},
    {"name": "Fiesole, Toscana, Italia", "aliases": ["fiesole"], "bbox": [11.25, 43.78, 11.35, 43.84]},
]
UTC = timezone.utc


def call(name, **args):
    return ModelTurn(tool_calls=[ToolCall(name=name, args=args)])


def text(t):
    return ModelTurn(text=t)


class Harness:
    def __init__(self, steps=(), models=("model-a",), **kwargs):
        self.provider = ScriptedProvider(steps)
        self.repo = kwargs.pop("job_repo", None) or InMemoryJobRepository()
        self.geocoder = StubGeocoder(GAZETTEER)
        self.sessions = SessionStore()
        self.sleeps = []
        self.agent = Agent(
            self.provider,
            models=list(models),
            job_repo=self.repo,
            geocode=self.geocoder,
            sessions=self.sessions,
            clock=lambda: FIXED_NOW,
            sleep=self.sleeps.append,
            **kwargs,
        )

    def add(self, *steps):
        self.provider.steps.extend(steps)

    def tool_results_seen(self, call_index):
        """ToolResults the model received on its call_index-th generate() call."""
        msgs = self.provider.calls[call_index].messages
        return [r for m in msgs if isinstance(m, ToolResults) for r in m.results]


# ---------------------------------------------------------------- dispatch

def test_launch_analysis_creates_job_and_feeds_result_back():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=90), relay_tool_results])
    res = h.agent.run("ndvi firenze ultimi 3 mesi", "s1")

    [job] = h.repo.all()
    assert res.job_id == str(job.id)
    assert res.bbox == pytest.approx([11.15, 43.72, 11.33, 43.83])
    assert job.algorithm == "ndvi" and job.provider == "element84-s2"
    assert job.end_date == FIXED_NOW
    assert job.end_date - job.start_date == timedelta(days=90)
    assert job.end_date.tzinfo is not None  # timezone-aware, no utcnow()
    assert job.params == {"resolution_m": 20, "geocoded_name": "Firenze, Toscana, Italia", "apply_cloud_mask": True}

    [tool_result] = h.tool_results_seen(1)
    assert tool_result.name == "launch_analysis"
    assert tool_result.response["job_id"] == res.job_id
    assert tool_result.response["days_back"] == 90
    assert res.text == tool_result.response["message"]
    assert [c.name for c in res.tool_calls] == ["launch_analysis"]
    assert res.model == "model-a"


def test_provider_receives_system_prompt_and_all_tools():
    h = Harness([text("ciao")])
    h.agent.run("ciao", "s1")
    rec = h.provider.calls[0]
    assert rec.system_prompt == (
        SYSTEM_PROMPT + "\nTODAY: 2026-01-15 (UTC). Use it to resolve 'today', 'yesterday', "
        "'this month' and dates given without a year."
    )
    assert rec.tool_names == [t.name for t in TOOL_SPECS]
    assert rec.messages == [UserMessage(text="ciao")]


def test_alias_and_default_window():
    h = Harness([call("launch_analysis", location="Pisa", algorithm="vegetazione"), relay_tool_results])
    h.agent.run("vegetazione pisa", "s1")
    [job] = h.repo.all()
    assert job.algorithm == "ndvi"
    assert job.end_date - job.start_date == timedelta(days=30)


@pytest.mark.parametrize("days", [90.0, "90"])
def test_days_back_number_types_are_accepted(days):
    h = Harness([call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=days), relay_tool_results])
    assert h.agent.run("x", "s1").job_id is not None
    [job] = h.repo.all()
    assert job.end_date - job.start_date == timedelta(days=90)


def test_days_back_null_means_default():
    h = Harness([call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=None), relay_tool_results])
    h.agent.run("x", "s1")
    [job] = h.repo.all()
    assert job.end_date - job.start_date == timedelta(days=30)


def test_list_capabilities():
    h = Harness([call("list_capabilities"), relay_tool_results])
    h.agent.run("cosa sai fare?", "s1")
    [r] = h.tool_results_seen(1)
    assert sorted(a["name"] for a in r.response["algorithms"]) == ["flood", "lisa", "mndwi", "ndvi", "ndwi"]


def test_multiple_calls_in_one_turn_all_dispatched():
    h = Harness([
        ModelTurn(tool_calls=[
            ToolCall("launch_analysis", {"location": "Firenze", "algorithm": "ndvi"}),
            ToolCall("launch_analysis", {"location": "Pisa", "algorithm": "flood"}),
        ]),
        relay_tool_results,
    ])
    res = h.agent.run("ndvi firenze e flood pisa", "s1")
    jobs = h.repo.all()
    assert len(jobs) == 2
    assert res.job_id == str(jobs[-1].id)  # last created job is reported
    assert len(h.tool_results_seen(1)) == 2


# ---------------------------------------------------------------- errors relayed

def test_unknown_tool_is_reported_to_model():
    h = Harness([call("delete_everything", confirm=True), relay_tool_results])
    res = h.agent.run("x", "s1")
    [r] = h.tool_results_seen(1)
    assert r.response == {"error": "Unknown function: delete_everything"}
    assert res.job_id is None
    assert res.text == "Unknown function: delete_everything"


@pytest.mark.parametrize(
    "args,fragment",
    [
        ({"location": "Firenze", "algorithm": "traffic"}, "Unknown algorithm 'traffic'"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": 400}, "exceeds maximum"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": 0}, "must be >= 1"),
        ({"location": "xqzvbnm", "algorithm": "ndvi"}, "Could not geocode"),
        ({"location": "  ", "algorithm": "ndvi"}, "Location is empty"),
    ],
)
def test_validation_errors_are_relayed_and_create_no_job(args, fragment):
    h = Harness([call("launch_analysis", **args), relay_tool_results])
    res = h.agent.run("x", "s1")
    [r] = h.tool_results_seen(1)
    assert fragment in r.response["error"]
    assert fragment in res.text
    assert res.job_id is None and h.repo.all() == []


def test_handler_exception_is_relayed():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi", resolution=5), relay_tool_results])
    h.agent.run("x", "s1")
    [r] = h.tool_results_seen(1)
    assert r.response["error"].startswith("Internal error in launch_analysis:")


def test_repository_failure_is_relayed():
    class BrokenRepo(InMemoryJobRepository):
        def create(self, **kwargs):
            raise RuntimeError("db down")

    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi"), relay_tool_results], job_repo=BrokenRepo())
    res = h.agent.run("x", "s1")
    assert res.text == "Failed to create job: db down"
    assert res.job_id is None


# ---------------------------------------------------------------- needs-user guard
#
# After a job-creating call is rejected for a reason only the user can resolve,
# no further job-creating call may execute in the same request.

LIMIT_730 = "days_back=730 exceeds maximum"


def test_window_rejected_at_730_then_retried_at_365_creates_no_job():
    # The exact live sequence of eval case it-refuse-window-2-anni.
    h = Harness([
        call("launch_analysis", location="Bologna", algorithm="flood", days_back=730),
        call("launch_analysis", location="Bologna", algorithm="flood", days_back=365),
        relay_tool_results,
    ])
    res = h.agent.run("alluvioni a Bologna negli ultimi 2 anni", "s1")

    assert h.repo.all() == [] and res.job_id is None
    first, second = res.tool_calls
    assert first.args["days_back"] == 730 and LIMIT_730 in first.result["error"]
    assert first.result["needs_user"] is True and "Only the user" in first.result["action_required"]
    assert second.args["days_back"] == 365
    assert second.result["blocked"] is True and second.result["error"] == first.result["error"]
    assert "NOT EXECUTED" in second.result["action_required"]
    assert h.geocoder.queries == []  # the blocked call never reached the geocoder
    [seen] = h.tool_results_seen(2)[1:]
    assert seen.response["blocked"] is True  # the model is told why
    assert LIMIT_730 in res.text


def test_second_call_in_the_same_round_is_blocked():
    h = Harness([
        ModelTurn(tool_calls=[
            ToolCall("launch_analysis", {"location": "Bologna", "algorithm": "flood", "days_back": 730}),
            ToolCall("launch_analysis", {"location": "Bologna", "algorithm": "flood", "days_back": 365}),
        ]),
        relay_tool_results,
    ])
    res = h.agent.run("alluvioni a Bologna negli ultimi 2 anni", "s1")
    assert h.repo.all() == [] and res.job_id is None
    assert [c.result.get("blocked", False) for c in res.tool_calls] == [False, True]


def test_rerun_is_blocked_too_after_a_needs_user_rejection():
    h = Harness([
        call("launch_analysis", location="Pisa", algorithm="flood"), relay_tool_results,
        call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=1825),
        call("rerun_with_different_algorithm", algorithm="ndvi"),
        relay_tool_results,
    ])
    h.agent.run("flood pisa", "s1")
    res = h.agent.run("ndvi stessa zona ultimi 5 anni", "s1")
    assert len(h.repo.all()) == 1 and res.job_id is None
    assert res.tool_calls[1].name == "rerun_with_different_algorithm"
    assert res.tool_calls[1].result["blocked"] is True


def test_no_previous_area_blocks_a_launch_on_an_invented_place():
    h = Harness([
        call("rerun_with_different_algorithm", algorithm="flood"),
        call("launch_analysis", location="Firenze", algorithm="flood"),
        relay_tool_results,
    ])
    res = h.agent.run("stessa zona ma con alluvione", "fresh-session")
    assert h.repo.all() == [] and res.job_id is None
    assert res.tool_calls[0].result["needs_user"] is True
    assert res.tool_calls[1].result["blocked"] is True
    assert res.text == "No previous analysis in this session. Specify a location."


def test_empty_location_blocks_an_invented_place():
    h = Harness([
        call("launch_analysis", location="", algorithm="ndvi"),
        call("launch_analysis", location="Firenze", algorithm="ndvi"),
        relay_tool_results,
    ])
    res = h.agent.run("fammi un'analisi ndvi", "s1")
    assert h.repo.all() == [] and res.tool_calls[1].result["blocked"] is True


@pytest.mark.parametrize(
    "bad,fixed,fragment",
    [
        ({"location": "Firenze", "algorithm": "salute colture"}, {"location": "Firenze", "algorithm": "ndvi"},
         "Unknown algorithm"),
        ({"location": "Fiorenze", "algorithm": "ndvi"}, {"location": "Firenze", "algorithm": "ndvi"},
         "Could not geocode"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": "three months"},
         {"location": "Firenze", "algorithm": "ndvi", "days_back": 90}, "whole number"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": 0},
         {"location": "Firenze", "algorithm": "ndvi", "days_back": 1}, "must be >= 1"),
        ({"location": "Firenze " + "x" * 200, "algorithm": "ndvi"}, {"location": "Firenze", "algorithm": "ndvi"},
         "too long"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": 60, "start_date": "2025-08-01",
          "end_date": "2025-09-30"},
         {"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-08-01", "end_date": "2025-09-30"},
         "not both"),
        ({"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-12-01"},
         {"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-12-01", "end_date": "2026-01-15"},
         "end_date is missing"),
        ({"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-12-01", "end_date": "2026-02-28"},
         {"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-12-01", "end_date": "2026-01-15"},
         "is in the future"),
        ({"location": "Firenze", "algorithm": "ndvi", "start_date": "01/08/2025", "end_date": "2025-08-31"},
         {"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-08-01", "end_date": "2025-08-31"},
         "YYYY-MM-DD"),
        ({"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-09-01", "end_date": "2025-09-31"},
         {"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-09-01", "end_date": "2025-09-30"},
         "not a real calendar day"),
        ({"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-09-30", "end_date": "2025-08-01"},
         {"location": "Firenze", "algorithm": "ndvi", "start_date": "2025-08-01", "end_date": "2025-09-30"},
         "is after end_date"),
    ],
    ids=["unknown-alias", "geocode-miss", "days-format", "days-zero", "location-too-long",
         "dates-and-days-back", "start-without-end", "end-in-future", "date-format", "date-not-a-day",
         "start-after-end"],
)
def test_model_fixable_rejection_allows_the_corrected_retry(bad, fixed, fragment):
    h = Harness([call("launch_analysis", **bad), call("launch_analysis", **fixed), relay_tool_results])
    res = h.agent.run("x", "s1")
    first, second = res.tool_calls
    assert fragment in first.result["error"]
    assert "needs_user" not in first.result and "blocked" not in second.result
    [job] = h.repo.all()
    assert res.job_id == str(job.id) and job.algorithm == "ndvi"


def test_non_job_tools_still_run_after_a_block():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=730),
        call("list_capabilities"),
        relay_tool_results,
    ])
    res = h.agent.run("x", "s1")
    assert "algorithms" in res.tool_calls[1].result


def test_block_lasts_for_one_request_only():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=730), relay_tool_results,
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=365), relay_tool_results,
    ])
    assert h.agent.run("ndvi firenze ultimi 2 anni", "s1").job_id is None
    # The user now asks for one year themselves: a new request, so it runs.
    assert h.agent.run("ok, allora l'ultimo anno", "s1").job_id is not None


def test_block_survives_a_retry_of_the_same_request():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=730),
        RateLimitError("429"),
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=365),
        relay_tool_results,
    ])
    res = h.agent.run("ndvi firenze ultimi 2 anni", "s1")
    assert h.repo.all() == [] and res.job_id is None
    assert res.tool_calls[-1].result["blocked"] is True


def test_round_cap_after_a_block_reports_the_limit():
    steps = [call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=d)
             for d in (730, 365, 300, 200, 100)]
    h = Harness(steps, max_tool_rounds=5)
    res = h.agent.run("x", "s1")
    assert h.repo.all() == []
    assert res.text.startswith(LIMIT_730)


def test_job_created_before_a_limit_is_kept_and_reported():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi"),
        call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=730),
        call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=365),
        relay_tool_results,
    ])
    res = h.agent.run("ndvi firenze, e pisa ultimi 2 anni", "s1")
    [job] = h.repo.all()
    assert res.job_id == str(job.id) and job.params["geocoded_name"].startswith("Firenze")


# ---------------------------------------------------------------- the period: days_back or calendar dates

def test_calendar_period_creates_the_job_on_exactly_those_days():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi",
                      start_date="2025-08-01", end_date="2025-09-30"), relay_tool_results])
    res = h.agent.run("ndvi florence, August to September 2025", "s1")
    [job] = h.repo.all()
    assert job.start_date == datetime(2025, 8, 1, tzinfo=UTC)
    assert job.end_date == datetime(2025, 9, 30, 23, 59, 59, 999999, tzinfo=UTC)
    [r] = h.tool_results_seen(1)
    assert r.response["date_range"] == "2025-08-01 to 2025-09-30"
    assert (r.response["start_date"], r.response["end_date"]) == ("2025-08-01", "2025-09-30")
    assert "days_back" not in r.response  # a calendar period is never described as "N days"
    assert "(2025-08-01 to 2025-09-30, 20m)" in r.response["message"]
    assert res.text == r.response["message"]


def test_look_back_result_keeps_days_back_and_states_the_dates():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=90), relay_tool_results])
    h.agent.run("ndvi florence, last 3 months", "s1")
    [r] = h.tool_results_seen(1)
    first = str((FIXED_NOW - timedelta(days=90)).date())
    assert r.response["days_back"] == 90
    assert (r.response["start_date"], r.response["end_date"]) == (first, "2026-01-15")
    assert r.response["date_range"] == f"{first} to 2026-01-15"


def test_calendar_period_ending_today_ends_at_the_clock_reading():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi",
                      start_date="2025-12-01", end_date="2026-01-15"), relay_tool_results])
    h.agent.run("ndvi florence from 1 December to today", "s1")
    [job] = h.repo.all()
    assert (job.start_date, job.end_date) == (datetime(2025, 12, 1, tzinfo=UTC), FIXED_NOW)


@pytest.mark.parametrize("same_round", [False, True], ids=["later-round", "same-round"])
@pytest.mark.parametrize(
    "first,retry,fragment",
    [
        ({"start_date": "2027-03-01", "end_date": "2027-03-31"}, {"days_back": 30}, "starts in the future"),
        ({"start_date": "2024-01-01", "end_date": "2025-12-31"}, {"days_back": 365}, "spans 730 days"),
        ({"start_date": "2024-01-01", "end_date": "2025-12-31"},
         {"start_date": "2025-01-01", "end_date": "2025-12-31"}, "spans 730 days"),
        ({"end_date": "2025-09-30"}, {"start_date": "2025-09-01", "end_date": "2025-09-30"}, "no start date"),
    ],
    ids=["future-then-days-back", "too-long-then-365", "too-long-then-shorter-dates", "no-start-then-invented"],
)
def test_needs_user_period_blocks_any_retry(first, retry, fragment, same_round):
    base = {"location": "Firenze", "algorithm": "ndvi"}
    calls = [ToolCall("launch_analysis", {**base, **first}), ToolCall("launch_analysis", {**base, **retry})]
    steps = [ModelTurn(tool_calls=calls)] if same_round else [ModelTurn(tool_calls=[c]) for c in calls]
    h = Harness([*steps, relay_tool_results])
    res = h.agent.run("x", "s1")
    rejected, blocked = (c.result for c in res.tool_calls)
    assert fragment in rejected["error"] and rejected["needs_user"] is True and "blocked" not in rejected
    assert blocked["blocked"] is True and blocked["error"] == rejected["error"]
    assert h.repo.all() == [] and res.job_id is None
    assert h.geocoder.queries == []  # neither call reached the geocoder
    assert "Period analysed" not in res.text


def test_one_clock_reading_per_request_tells_the_model_today_and_sets_every_window():
    reads = []

    def clock():
        reads.append(1)
        return datetime(2026, 11, 27, 9, 30, tzinfo=UTC) + timedelta(hours=len(reads))

    h = Harness([
        ModelTurn(tool_calls=[
            ToolCall("launch_analysis", {"location": "Firenze", "algorithm": "ndvi"}),
            ToolCall("launch_analysis", {"location": "Pisa", "algorithm": "ndvi",
                                         "start_date": "2026-11-01", "end_date": "2026-11-27"}),
        ]),
        relay_tool_results,
    ])
    h.agent.clock = clock
    h.agent.run("ndvi florence, and pisa since the start of the month", "s1")
    now = datetime(2026, 11, 27, 10, 30, tzinfo=UTC)
    assert len(reads) == 1
    assert [j.end_date for j in h.repo.all()] == [now, now]
    assert all(c.system_prompt.endswith(
        "TODAY: 2026-11-27 (UTC). Use it to resolve 'today', 'yesterday', 'this month' and dates "
        "given without a year.") for c in h.provider.calls)


def test_today_told_to_the_model_is_the_utc_date():
    h = Harness([text("ok")])
    h.agent.clock = lambda: datetime(2026, 11, 28, 0, 30, tzinfo=timezone(timedelta(hours=2)))
    h.agent.run("hello", "s1")
    assert "\nTODAY: 2026-11-27 (UTC)." in h.provider.calls[0].system_prompt


def test_launch_schema_takes_calendar_dates_and_states_no_maximum():
    spec = next(t for t in TOOL_SPECS if t.name == "launch_analysis")
    props = spec.parameters["properties"]
    assert spec.parameters["required"] == ["location", "algorithm"]
    assert props["start_date"]["type"] == props["end_date"]["type"] == "string"
    assert "YYYY-MM-DD" in props["start_date"]["description"]
    assert "YYYY-MM-DD" in props["end_date"]["description"]
    assert "Never turn a calendar period" in props["days_back"]["description"]
    assert "today's date" in props["end_date"]["description"]
    # The limit is the tool's to check: the model must not pre-shorten a period.
    assert "maximum" not in str(spec.parameters) and "maximum" not in SYSTEM_PROMPT


def test_prompt_rules_for_calendar_periods():
    flat = " ".join(SYSTEM_PROMPT.split())  # rules are wrapped with continuation lines
    rule10 = flat.split(" 10. ")[1].split(" 11. ")[0]
    assert "Never convert a calendar period into days_back" in rule10
    assert "most recent occurrence that is not in the future" in rule10
    assert "pass start_date and end_date" in rule10 and "needs_user" in rule10
    rule6 = flat.split(" 6. ")[1].split(" 7. ")[0]
    assert "start and end dates from the tool result (date_range)" in rule6
    assert "never only as a number of days" in rule6


# The reply states the period that ran (code-written when the model's text lacks the dates).

def test_reply_states_the_period_when_the_model_does_not():
    # Replay of the live failure of 2026-11-27, on the test clock: "August to September"
    # launched as days_back=60, reply "Temporal Window: 60 days". Code cannot tell this
    # first call from a request for the last 60 days; it makes sure the reply says so.
    h = Harness([call("launch_analysis", location="Fiesole", algorithm="lisa", days_back=60),
                 text("LISA analysis launched for Fiesole. Temporal Window: 60 days.")])
    res = h.agent.run("Find spatial outliers in vegetation around Fiesole, August to September 2026", "s1")
    assert res.job_id is not None
    assert res.text == ("LISA analysis launched for Fiesole. Temporal Window: 60 days.\n\n"
                        "Period analysed: 2025-11-16 to 2026-01-15.")


def test_reply_with_both_dates_is_left_unchanged():
    reply = "LISA analysis launched for Fiesole, from 2025-08-01 to 2025-09-30."
    h = Harness([call("launch_analysis", location="Fiesole", algorithm="lisa",
                      start_date="2025-08-01", end_date="2025-09-30"), text(reply)])
    assert h.agent.run("x", "s1").text == reply


def test_dates_inside_a_stripped_marker_do_not_count():
    h = Harness([call("launch_analysis", location="Fiesole", algorithm="lisa",
                      start_date="2025-08-01", end_date="2025-09-30"),
                 text("Done [JOB_CREATED:2025-08-01 to 2025-09-30]")])
    assert h.agent.run("x", "s1").text == "Done\n\nPeriod analysed: 2025-08-01 to 2025-09-30."


@pytest.mark.parametrize("steps", [
    [text("hello")],
    [call("launch_analysis", location="Firenze", algorithm="ndvi",
          start_date="2027-03-01", end_date="2027-03-31"), relay_tool_results],
], ids=["no-call", "rejected-period"])
def test_no_job_no_period_sentence(steps):
    res = Harness(steps).agent.run("x", "s1")
    assert res.job_id is None and "Period analysed" not in res.text


def test_round_cap_text_with_a_job_states_the_period():
    steps = [call("launch_analysis", location="Firenze", algorithm="ndvi")]
    steps += [call("list_capabilities") for _ in range(4)]
    res = Harness(steps, max_tool_rounds=5).agent.run("x", "s1")
    assert res.job_id is not None
    assert res.text == "Analysis request is being processed.\n\nPeriod analysed: 2025-12-16 to 2026-01-15."


def test_get_job_status_reports_the_calendar_days():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi",
                      start_date="2025-08-01", end_date="2025-09-30"), text("ok"),
                 call("get_job_status"), relay_tool_results])
    h.agent.run("ndvi florence, August to September 2025", "s1")
    status = h.agent.run("is it done?", "s1")
    assert status.tool_calls[0].result["time_window"] == "2025-08-01 to 2025-09-30"


# ---------------------------------------------------------------- loop limits

def test_max_tool_rounds_cap():
    h = Harness([call("list_capabilities") for _ in range(10)], max_tool_rounds=5)
    res = h.agent.run("x", "s1")
    assert len(h.provider.calls) == 5
    assert len(res.tool_calls) == 5
    assert res.text == "Analysis request is being processed."


def test_no_candidate():
    h = Harness([None])
    assert h.agent.run("x", "s1").text == "No response from AI model."


def test_job_marker_is_stripped():
    h = Harness([text("Fatto [JOB_CREATED:abc]")])
    assert h.agent.run("x", "s1").text == "Fatto"


# ---------------------------------------------------------------- fallback & retry

def test_daily_quota_moves_to_next_model_without_waiting():
    h = Harness([RateLimitError("429 PerDay", daily_quota=True), text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "b"]
    assert h.sleeps == []
    assert res.text == "ok" and res.model == "b"


def test_rate_limit_retries_with_backoff_then_succeeds():
    h = Harness([RateLimitError("429"), RateLimitError("429"), text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "a", "a"]
    assert h.sleeps == [5, 10]
    assert res.text == "ok"


def test_rate_limit_exhausts_retries_then_falls_back():
    h = Harness([RateLimitError("429")] * 3 + [text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "a", "a", "b"]
    assert h.sleeps == [5, 10]  # no pointless sleep after the last attempt
    assert res.model == "b"


def test_other_provider_error_falls_back_immediately():
    h = Harness([ProviderError("500 boom"), text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "b"]
    assert h.sleeps == [] and res.text == "ok"


def test_all_models_exhausted():
    h = Harness([ProviderError("down"), ProviderError("down")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert res.text == "AI service temporarily unavailable. Please retry in a minute."
    assert "down" in res.error
    assert res.job_id is None


def test_failure_after_job_creation_does_not_duplicate_the_job():
    h = Harness(
        [call("launch_analysis", location="Firenze", algorithm="ndvi"), RateLimitError("429"), text("never")],
        models=["a", "b"],
    )
    res = h.agent.run("ndvi firenze", "s1")
    assert len(h.repo.all()) == 1
    assert len(h.provider.calls) == 2  # no retry, no fallback
    assert res.job_id == str(h.repo.all()[0].id)
    assert res.text.startswith("Analysis 'NDVI' queued for Firenze")
    assert "Period analysed" not in res.text  # the launch message already states the dates


def test_requires_a_model():
    with pytest.raises(ValueError):
        Agent(ScriptedProvider(), models=[], job_repo=InMemoryJobRepository())


# ---------------------------------------------------------------- input handling

def test_input_is_sanitised_before_reaching_the_model():
    h = Harness([text("ok")])
    raw = "ndvi\x00 fi\x07renze\n\tok" + "x" * 3000
    h.agent.run(raw, "s1")
    sent = h.provider.calls[0].messages[0].text
    assert "\x00" not in sent and "\x07" not in sent
    assert sent.startswith("ndvi firenze\n\tok")
    assert len(sent) == 2000


@pytest.mark.parametrize("raw", ["", "   ", "\x00\x01\x02"])
def test_empty_input_never_reaches_the_model(raw):
    h = Harness([])
    assert h.agent.run(raw, "s1").text == "Please enter a message."
    assert h.provider.calls == []


def test_each_request_is_stateless_towards_the_model():
    h = Harness([text("one"), text("two")])
    h.agent.run("first", "s1")
    h.agent.run("second", "s1")
    assert h.provider.calls[1].messages == [UserMessage(text="second")]


# ---------------------------------------------------------------- session / rerun

def test_rerun_uses_session_context():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi"), relay_tool_results,
        call("rerun_with_different_algorithm", algorithm="water"), relay_tool_results,
    ])
    first = h.agent.run("ndvi firenze", "s1")
    second = h.agent.run("stessa zona, acqua", "s1")
    jobs = h.repo.all()
    assert [j.algorithm for j in jobs] == ["ndvi", "ndwi"]
    assert jobs[1].bbox == jobs[0].bbox
    assert second.job_id != first.job_id
    ctx = h.sessions.get_or_create("s1").context
    assert ctx["last_algorithm"] == "ndwi" and ctx["last_job_id"] == second.job_id


def test_rerun_without_algorithm_reuses_previous():
    h = Harness([
        call("launch_analysis", location="Pisa", algorithm="flood"), relay_tool_results,
        call("rerun_with_different_algorithm"), relay_tool_results,
    ])
    h.agent.run("flood pisa", "s1")
    h.agent.run("ripeti", "s1")
    assert [j.algorithm for j in h.repo.all()] == ["flood", "flood"]


def test_rerun_is_isolated_per_session():
    h = Harness([
        call("launch_analysis", location="Pisa", algorithm="flood"), relay_tool_results,
        call("rerun_with_different_algorithm", algorithm="ndvi"), relay_tool_results,
    ])
    h.agent.run("flood pisa", "s1")
    res = h.agent.run("same area, ndvi", "other-session")
    assert res.text == "No previous analysis in this session. Specify a location."
    assert len(h.repo.all()) == 1


def test_rerun_without_session_id():
    h = Harness([call("rerun_with_different_algorithm", algorithm="ndvi"), relay_tool_results])
    assert h.agent.run("same area", "").text == "No session context. Please specify a location."


# ---------------------------------------------------------------- job status

def test_get_job_status_variants():
    h = Harness()
    job = h.repo.create(algorithm="flood", provider="p", bbox=[0, 0, 1, 1],
                        start_date=FIXED_NOW, end_date=FIXED_NOW, params={})
    job.status, job.result = "completed", {"tile_url": "http://t", "summary": {"a": 1}, "other": 2}
    from eo_brain.agent.brain import _RequestContext

    ctx = _RequestContext(session_id="s1")
    assert h.agent._get_job_status(ctx, "not-a-uuid") == {"error": "Invalid job ID: not-a-uuid"}
    missing = str(UUID(int=999))
    assert h.agent._get_job_status(ctx, missing) == {"error": f"Job {missing} not found."}
    found = h.agent._get_job_status(ctx, str(job.id))
    assert found["status"] == "completed"
    assert found["result"] == {
        "tile_url": "http://t", "summary": {"a": 1},
        "coverage": {"note": "coverage was not recorded for this analysis (it ran before coverage reporting)"},
    }


def test_get_job_status_reports_coverage_facts_only():
    h = Harness()
    job = h.repo.create(algorithm="ndvi", provider="p", bbox=[0, 0, 1, 1],
                        start_date=FIXED_NOW - timedelta(days=30), end_date=FIXED_NOW,
                        params={"geocoded_name": "Casentino"})
    coverage = {"coverage_pct": 7.2, "low_coverage": True, "uncovered_masked_pct": 0.3,
                "uncovered_no_data_pct": 92.4, "scenes_found": 27, "scenes_used": 1,
                "area_px": 380172, "selection": {"scene_ids": ["x"]}}
    job.status, job.result = "completed", {"tile_url": "t", "summary": {}, "coverage": coverage}
    job.message = "Completed with partial coverage: ..."
    from eo_brain.agent.brain import _RequestContext

    found = h.agent._get_job_status(_RequestContext(session_id="s1"), str(job.id))
    assert found["location"] == "Casentino"
    assert found["time_window"] == f"{(FIXED_NOW - timedelta(days=30)).date()} to {FIXED_NOW.date()}"
    assert found["message"].startswith("Completed with partial coverage")
    assert found["result"]["coverage"] == {
        "coverage_pct": 7.2, "low_coverage": True, "uncovered_masked_pct": 0.3,
        "uncovered_no_data_pct": 92.4, "scenes_found": 27, "scenes_used": 1,
    }


def test_get_job_status_without_id_uses_the_sessions_latest_job():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi"), text("ok"),
                 call("get_job_status"), relay_tool_results])
    launched = h.agent.run("ndvi firenze", "s1")
    status = h.agent.run("perché la mappa copre solo una striscia?", "s1")
    assert launched.job_id and status.job_id is None
    [inv] = status.tool_calls
    assert inv.name == "get_job_status" and inv.result["job_id"] == launched.job_id


def test_get_job_status_without_id_and_without_a_job_says_so():
    from eo_brain.agent.brain import _RequestContext

    h = Harness()
    out = h.agent._get_job_status(_RequestContext(session_id="fresh"))
    assert "No analysis was launched in this conversation" in out["error"]
    assert "query_database" in out["error"]


def test_prompt_forbids_invented_causes():
    rule = brain.SYSTEM_PROMPT.split("13.")[1]
    assert "only from tool results" in rule and "never offer causes" in rule
    assert "get_job_status" in rule and "query_database" in rule


def test_prompt_states_a_threshold_with_every_figure_that_depends_on_one():
    flat = " ".join(SYSTEM_PROMPT.split())
    rule = flat.split(" 14. ")[1]
    assert "stated with that threshold, from the threshold column of the rows" in rule
    assert "if the user named none, say it is the app's default" in rule
    assert "NDVI 0.65 for dense green vegetation" in rule and "0 for water with NDWI or MNDWI" in rule
    assert "A NULL area is not zero" in rule
    # the question reaches the data agent with the user's threshold, and the routing says where it goes
    spec = next(t for t in TOOL_SPECS if t.name == "query_database")
    assert "how much green" in spec.description and "with any threshold the user gave" in spec.description
    assert "how much green or how much water) with query_database" in flat


def test_prompt_states_zone_counts_with_their_size_and_flood_water_as_water_seen():
    flat = " ".join(SYSTEM_PROMPT.split())
    rule = flat.split(" 15. ")[1].split(" 16. ")[0]
    assert "state a number of zones (anomalies, flooded areas) with the minimum zone size" in rule
    assert "the number of smaller patches left out" in rule
    assert "permanent water (sea, lakes, rivers) included" in rule
    assert "never newly flooded land or a change" in rule
    assert "A NULL zone size, area or share is not available, never zero" in rule
    # routing: such questions are about stored results
    spec = next(t for t in TOOL_SPECS if t.name == "query_database")
    assert "for flood and LISA, the exact hectares and share of each severity" in spec.description
    assert "for flood and LISA, the exact hectares and share of each severity" in flat


def test_prompt_keeps_zone_counts_and_class_areas_apart():
    """Live reply of 2026-11-27 (LISA, Fiesole): "534 anomaly zones (of at least 1,600 m2) were
    mapped, covering 394.28 hectares". Every number was in the rows, but 394.28 ha is the area of
    the anomaly classes, small patches included; the 534 zones alone cover 143.44 ha."""
    flat = " ".join(SYSTEM_PROMPT.split())
    rule = flat.split(" 16. ")[1]
    assert "zones and areas are two measures; never merge them" in rule
    assert "The hectares and share of a severity, of the anomalies or of the water are the exact area" in rule
    assert "every pixel, the smaller patches included" in rule
    assert "so more than the mapped zones cover" in rule
    assert "with the zones as a separate count, also in a list per severity" in rule
    # the two wordings of the live reply, as what not to write
    assert 'never as what the zones cover ("3 zones covering 12.64 ha", "Alert: 3 zones | 12.64 ha")' in rule
    # the area of the zones is said only of a column the data agent names as such
    assert "Zones cover only an area the rows give as theirs (mapped_zones_area_ha)" in rule
    # the capability list and the tool description name them as two measures, not as one figure
    spec = next(t for t in TOOL_SPECS if t.name == "query_database")
    assert "and, as a separate measure, the number of mapped zones" in spec.description
    assert "and, as a separate measure, how many zones were mapped" in flat
    assert "share of each severity and how many zones" not in flat
    assert "share of each severity and the number of mapped zones" not in spec.description


# ---------------------------------------------------------------- module entry point

def test_chat_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(brain, "_default_agent", None)
    assert brain.chat("ndvi firenze", "s1") == {
        "text": "AI agent is not configured. Set GEMINI_API_KEY.", "job_id": None, "bbox": None, "query": None,
    }


def test_chat_uses_injected_default_agent(monkeypatch):
    h = Harness([text("ciao")])
    monkeypatch.setattr(brain, "_default_agent", h.agent)
    assert brain.chat("ciao", "s1") == {"text": "ciao", "job_id": None, "bbox": None, "query": None}


def test_launch_reports_a_geocoder_fallback():
    from eo_brain.agent.geocoder import GeocodingResult

    def geocode(q):
        return GeocodingResult(name="Casentino, Arezzo, Toscana, Italia", lat=43.7, lon=11.8,
                               bbox=(11.6, 43.55, 11.98, 43.88), source="nominatim_bbox",
                               matched_query="Casentino", fallback_from=q)

    h = Harness([call("launch_analysis", location="vallata del Casentino", algorithm="ndvi"), relay_tool_results])
    h.agent.geocode = geocode
    res = h.agent.run("ndvi vallata del Casentino", "s1")
    [inv] = res.tool_calls
    assert inv.result["location"] == "Casentino, Arezzo, Toscana, Italia"
    assert "'vallata del Casentino' was not found as written" in inv.result["geocoding_note"]
    assert "used Casentino, Arezzo, Toscana, Italia" in inv.result["message"]


# ---------------------------------------------------------------- which place (recorded Nominatim responses)

from pathlib import Path  # noqa: E402

from eo_brain.agent.geocoder import GeocodingResult  # noqa: E402
from evals.stubs import load_recorded_searches  # noqa: E402

RECORDED = load_recorded_searches(Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "nominatim_recorded.json")
REGION = "Casentino, Arezzo, Tuscany, Italy"
VILLAGE = "Casentino, Sant'Eusanio Forconese, L'Aquila, Abruzzo, 67040, Italy"


def place_harness(*steps):
    h = Harness(list(steps))
    h.geocoder = StubGeocoder(GAZETTEER, RECORDED)
    h.agent.geocode = h.geocoder
    return h


def ask_casentino(h, session="s1"):
    h.add(call("launch_analysis", location="Casentino", algorithm="vegetation", days_back=60), relay_tool_results)
    return h.agent.run("ndvi Casentino ultimi 60 giorni", session)


def test_ambiguous_place_is_a_question_with_candidates_and_no_job():
    h = place_harness()
    res = ask_casentino(h)
    [inv] = res.tool_calls
    assert res.job_id is None and h.repo.all() == []
    assert inv.result["needs_user"] is True
    assert inv.result["candidates"] == [
        {"location": VILLAGE, "kind": "village", "in": "L'Aquila, Abruzzo, Italy"},
        {"location": REGION, "kind": "region", "in": "Arezzo, Tuscany, Italy"},
    ]
    assert "Do not pick one yourself" in inv.result["action_required"]
    assert "location' text exactly" in inv.result["action_required"]
    pending = h.sessions.get_or_create("s1").get_context("pending_place")
    assert pending["algorithm"] == "ndvi" and pending["days_back"] == 60
    assert [c.label for c in pending["candidates"]] == [VILLAGE, REGION]


def test_model_cannot_pick_a_candidate_itself_in_the_same_turn():
    h = place_harness(
        call("launch_analysis", location="Casentino", algorithm="ndvi"),
        call("launch_analysis", location=REGION, algorithm="ndvi"),
        text("Ho scelto io."),
    )
    res = h.agent.run("ndvi Casentino", "s1")
    assert res.job_id is None and h.repo.all() == []
    assert res.tool_calls[1].result["blocked"] is True


@pytest.mark.parametrize("label", [REGION, VILLAGE])
def test_the_answer_resolves_to_exactly_the_listed_candidate(label):
    h = place_harness()
    asked = ask_casentino(h)
    listed = {c["location"]: c for c in asked.tool_calls[0].result["candidates"]}
    searches = list(h.geocoder.searches)
    h.add(call("launch_analysis", location=label, algorithm="ndvi", days_back=60), relay_tool_results)
    res = h.agent.run("quello giusto", "s1")
    [job] = h.repo.all()
    assert res.job_id == str(job.id) and job.params["geocoded_name"] == label
    assert h.geocoder.searches == searches                  # pinned candidate: no new lookup
    expected = h.geocoder._resolver.resolve(label).bbox     # what that label names on its own
    assert job.bbox == pytest.approx(list(expected))
    assert label in listed
    assert h.sessions.get_or_create("s1").get_context("pending_place") is None


def test_an_answer_naming_the_place_and_its_province_picks_that_one():
    h = place_harness()
    ask_casentino(h)
    searches = list(h.geocoder.searches)
    h.add(call("launch_analysis", location="Casentino, Arezzo", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("Casentino in provincia di Arezzo", "s1")
    assert res.job_id is not None and h.repo.all()[0].params["geocoded_name"] == REGION
    assert h.geocoder.searches == searches


def test_repeating_the_ambiguous_name_asks_again():
    h = place_harness()
    ask_casentino(h)
    h.add(call("launch_analysis", location="Casentino", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("Casentino", "s1")
    assert res.job_id is None and res.tool_calls[0].result["candidates"]


def test_context_note_after_a_question_is_shown_once():
    h = place_harness()
    ask_casentino(h)
    h.add(text("Quale?"), text("Ok."))
    h.agent.run("quello in Toscana", "s1")
    h.agent.run("grazie", "s1")
    first = h.provider.calls[-2].messages[0].text
    assert first.startswith("[Context from the app, not written by the user")
    assert f"1) {VILLAGE} (village); 2) {REGION} (region)" in first
    assert "algorithm 'ndvi' and days_back 60" in first
    assert first.endswith("User message: quello in Toscana")
    assert h.provider.calls[-1].messages == [UserMessage(text="grazie")]


def ask_casentino_dates(h, session="s1"):
    h.add(call("launch_analysis", location="Casentino", algorithm="ndvi",
               start_date="2025-08-01", end_date="2025-09-30"), relay_tool_results)
    return h.agent.run("ndvi Casentino, August to September 2025", session)


def test_place_question_remembers_a_calendar_period_and_the_answer_runs_on_it():
    h = place_harness()
    asked = ask_casentino_dates(h)
    assert asked.job_id is None and asked.tool_calls[0].result["candidates"]
    pending = h.sessions.get_or_create("s1").get_context("pending_place")
    assert (pending["days_back"], pending["start_date"], pending["end_date"]) == (None, "2025-08-01", "2025-09-30")

    h.add(call("launch_analysis", location=REGION, algorithm="ndvi"), relay_tool_results)  # no period
    res = h.agent.run("the one in the province of Arezzo", "s1")
    note = h.provider.calls[-2].messages[0].text
    assert "(ndvi, 2025-08-01 to 2025-09-30)" in note
    assert "algorithm 'ndvi', start_date 2025-08-01 and end_date 2025-09-30" in note
    [job] = h.repo.all()
    assert res.job_id == str(job.id) and job.params["geocoded_name"] == REGION
    assert job.start_date == datetime(2025, 8, 1, tzinfo=UTC)
    assert job.end_date == datetime(2025, 9, 30, 23, 59, 59, 999999, tzinfo=UTC)
    assert "days_back" not in res.tool_calls[0].result


@pytest.mark.parametrize(
    "own,start,end",
    [({"days_back": 14}, FIXED_NOW - timedelta(days=14), FIXED_NOW),
     ({"start_date": "2025-07-01", "end_date": "2025-07-31"},
      datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 7, 31, 23, 59, 59, 999999, tzinfo=UTC))],
    ids=["days-back", "dates"],
)
def test_an_answer_with_its_own_period_uses_it(own, start, end):
    h = place_harness()
    ask_casentino_dates(h)
    h.add(call("launch_analysis", location=REGION, algorithm="ndvi", **own), relay_tool_results)
    h.agent.run("the Arezzo one, but another period", "s1")
    [job] = h.repo.all()
    assert (job.start_date, job.end_date) == (start, end)


def test_the_answer_without_a_period_reapplies_the_remembered_look_back_window():
    h = place_harness(call("launch_analysis", location="Casentino", algorithm="ndvi", days_back=90),
                      relay_tool_results)
    h.agent.run("ndvi Casentino, last 3 months", "s1")
    assert h.sessions.get_or_create("s1").get_context("pending_place")["days_back"] == 90
    later = FIXED_NOW + timedelta(days=1)
    h.agent.clock = lambda: later
    h.add(call("launch_analysis", location=REGION, algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("the Arezzo one", "s1")
    [job] = h.repo.all()
    assert (job.start_date, job.end_date) == (later - timedelta(days=90), later)
    assert res.tool_calls[0].result["days_back"] == 90


def test_a_remembered_calendar_period_is_checked_again_against_the_new_clock():
    h = place_harness(call("launch_analysis", location="Casentino", algorithm="ndvi",
                           start_date="2025-12-01", end_date="2026-01-15"), relay_tool_results)
    h.agent.run("ndvi Casentino from 1 December to today", "s1")
    h.agent.clock = lambda: FIXED_NOW + timedelta(days=1)
    h.add(call("launch_analysis", location=REGION, algorithm="ndvi"), relay_tool_results)
    h.agent.run("the Arezzo one", "s1")
    [job] = h.repo.all()
    # The period asked for ended on 2026-01-15; the next day it ends at the end of that day.
    assert job.end_date == datetime(2026, 1, 15, 23, 59, 59, 999999, tzinfo=UTC)


def test_valley_phrasing_launches_on_the_region_and_says_how_it_was_found():
    h = place_harness(call("launch_analysis", location="vallata del Casentino", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("Per favore, trova NDVI nella vallata del Casentino.", "s1")
    [inv] = res.tool_calls
    assert inv.result["location"] == REGION
    assert "searched for 'Casentino' instead and used " + REGION + " (region)" in inv.result["geocoding_note"]
    assert "Location: 'vallata del Casentino' not found as written" in inv.result["message"]


def test_region_asked_and_only_a_city_found_is_a_question():
    h = place_harness(call("launch_analysis", location="piana di Pisa", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("NDVI sulla piana di Pisa", "s1")
    [inv] = res.tool_calls
    assert res.job_id is None and inv.result["needs_user"]
    assert inv.result["candidates"] == [{"location": "Pisa, Toscana, Italia", "kind": "city", "in": "Toscana, Italia"}]
    assert "No region named as in 'piana di Pisa'" in inv.result["error"]


def test_a_non_area_last_resort_is_reported():
    def geocode(q):
        return GeocodingResult(name="Via Roma, Pisa", lat=43.7, lon=10.4, bbox=(10.39, 43.69, 10.41, 43.71),
                               source="nominatim_bbox", kind="road", area_like=False)

    h = Harness([call("launch_analysis", location="Via Roma, Pisa", algorithm="ndvi"), relay_tool_results])
    h.agent.geocode = geocode
    [inv] = h.agent.run("ndvi via roma pisa", "s1").tool_calls
    assert "Via Roma, Pisa is a road, not a town, region or natural area" in inv.result["geocoding_note"]
    assert "the analysis covers its bounding box" in inv.result["message"]


def test_place_lookups_per_request_are_bounded():
    h = place_harness(
        call("launch_analysis", location="Xyzzy", algorithm="ndvi"),
        call("launch_analysis", location="Plugh", algorithm="ndvi"),
        call("launch_analysis", location="Firenze", algorithm="ndvi"),
        text("..."),
    )
    res = h.agent.run("ndvi xyzzy", "s1")
    assert h.geocoder.queries == ["Xyzzy", "Plugh"]
    third = res.tool_calls[2].result
    assert third["needs_user"] and "could not be resolved in 2 attempts" in third["error"]
    assert res.job_id is None


def test_rerun_reuses_the_stored_area_without_a_lookup():
    h = place_harness(
        call("launch_analysis", location="vallata del Casentino", algorithm="ndvi"), relay_tool_results,
        call("rerun_with_different_algorithm", algorithm="water"), relay_tool_results,
    )
    h.agent.run("ndvi vallata del Casentino", "s1")
    queries = list(h.geocoder.queries)
    res = h.agent.run("stessa zona, acqua", "s1")
    first, second = h.repo.all()
    assert h.geocoder.queries == queries == ["vallata del Casentino"]
    assert second.bbox == first.bbox and second.params["geocoded_name"] == REGION
    assert "geocoding_note" not in res.tool_calls[0].result


def test_location_parameter_asks_for_commas_and_exact_candidates():
    spec = next(t for t in TOOL_SPECS if t.name == "launch_analysis")
    desc = spec.parameters["properties"]["location"]["description"]
    assert "separated by commas" in desc and "'Casentino, Arezzo, Italia'" in desc
    assert "candidate's location text exactly" in desc


def test_prompt_says_to_list_candidates_and_not_choose():
    assert "If it lists candidate places, show every candidate" in SYSTEM_PROMPT
    assert "never choose one yourself" in SYSTEM_PROMPT
