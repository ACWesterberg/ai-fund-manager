from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("launchd", ROOT / "deploy" / "macos" / "launchd.py")
launchd = importlib.util.module_from_spec(_spec)
sys.modules["launchd"] = launchd  # dataclasses look their module up
_spec.loader.exec_module(launchd)

MAC = Path("/Users/alexander/ai-fund-manager")
STUDIO = Path("/Users/alexander/ai-studio/.venv/bin/studio")


@pytest.fixture
def plists():
    return launchd.build(MAC, ROOT / "deploy" / "cron.example", STUDIO)


def test_every_crontab_line_and_unit_becomes_one_job(plists):
    table = (ROOT / "deploy" / "cron.example").read_text() + launchd.managed_block(
        ROOT / "deploy" / "cron.example")
    cron = [ln for ln in table.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    commands = {launchd.CD.sub("", ln.split(None, 5)[5]) for ln in cron}
    jobs = [p for p in plists.values() if "StartCalendarInterval" in p]
    assert {p["ProgramArguments"][-1] for p in jobs} == commands
    assert {"com.fundmgr.bot", "com.fundmgr.web", "com.fundmgr.global-web"} <= set(plists)
    for label, plist in plists.items():
        assert plist["Label"] == label
        plistlib.dumps(plist)  # launchd can read it


def test_times_match_the_crontab(plists):
    stops = plists["com.fundmgr.stops"]["StartCalendarInterval"]
    assert len(stops) == 4 * 9 * 5
    assert {d["Weekday"] for d in stops} == {1, 2, 3, 4, 5}
    assert {d["Hour"] for d in stops} == set(range(9, 18))
    assert plists["com.fundmgr.weekly"]["StartCalendarInterval"] == [
        {"Minute": 30, "Hour": 9, "Weekday": 1}]
    assert plists["com.fundmgr.optimize"]["StartCalendarInterval"] == [
        {"Minute": 0, "Hour": 2, "Weekday": 0}]
    # Three crontab lines, one command: one job at all three times.
    backup = plists["com.fundmgr.backup"]["StartCalendarInterval"]
    assert sorted(tuple(sorted(d.items())) for d in backup) == [
        (("Hour", 3), ("Minute", 0)), (("Hour", 11), ("Minute", 0), ("Weekday", 1)),
        (("Hour", 18), ("Minute", 0), ("Weekday", 1))]
    assert plists["com.fundmgr.optimizer_watch"]["StartCalendarInterval"] == [
        {"Minute": m} for m in (0, 15, 30, 45)]


def test_jobs_run_in_the_mac_checkout_under_studio_power_run(plists):
    for plist in plists.values():
        assert plist["WorkingDirectory"] == str(MAC)
        if "StartCalendarInterval" in plist:
            assert plist["ProgramArguments"][:4] == [str(STUDIO), "power", "run", "--"]
            assert "/home/alexander" not in plist["ProgramArguments"][-1]
            assert plist["EnvironmentVariables"]["PATH"].startswith(f"{MAC}/.venv/bin:")


def test_services_keep_the_units_commands_with_the_new_root(plists):
    web = plists["com.fundmgr.global-web"]
    assert web["ProgramArguments"] == [f"{MAC}/.venv/bin/uvicorn", "fundmgr.web.app:app",
                                       "--host", "0.0.0.0", "--port", "8011"]
    assert web["EnvironmentVariables"]["FUND_CONFIG"] == f"{MAC}/config/config_global.yaml"
    assert web["KeepAlive"] == {"SuccessfulExit": False} and web["RunAtLoad"] is True
    assert plists["com.fundmgr.bot"]["ProgramArguments"][-2:] == ["-m", "fundmgr.notify.telegram_bot"]


@pytest.mark.parametrize(("spec", "name", "want"), [
    ("*", "hour", None), ("*/20", "minute", [0, 20, 40]), ("1-5", "weekday", [1, 2, 3, 4, 5]),
    ("7", "weekday", [0]), ("9,11,14", "hour", [9, 11, 14]), ("0-10/5", "minute", [0, 5, 10]),
])
def test_cron_fields(spec, name, want):
    assert launchd.expand(spec, name) == want


def test_lines_launchd_cannot_express_are_refused():
    with pytest.raises(ValueError, match="both"):
        launchd.parse_crontab("0 9 1 * 1 echo hi\n")
    with pytest.raises(ValueError, match="range"):
        launchd.parse_crontab("0 25 * * * echo hi\n")
    with pytest.raises(ValueError, match="unsupported"):
        launchd.parse_crontab("@reboot echo hi\n")


def test_the_jobs_deploys_manage_in_cron_are_launchd_jobs_on_the_mac(plists):
    by_command = {p["ProgramArguments"][-1]: label for label, p in plists.items()
                  if "StartCalendarInterval" in p}
    for sub in ("account-sync --all", "account-plan", "earnings-watch", "report-check",
                "montrose-import --auto --notify"):
        assert any(f"bin/fund {sub}" in c for c in by_command), sub
    report = next(label for c, label in by_command.items() if "fund report-check" in c)
    assert sorted((d["Hour"], d["Minute"]) for d in plists[report]["StartCalendarInterval"]) == [
        (8, 45), (8, 45), (8, 45), (8, 45), (8, 45), (18, 30), (18, 30), (18, 30), (18, 30), (18, 30)]
    # Shared log files still give distinct, stable names.
    assert {"com.fundmgr.accounts", "com.fundmgr.accounts-account-plan",
            "com.fundmgr.montrose", "com.fundmgr.montrose-montrose-import"} <= set(plists)
