#!/usr/bin/env python3
"""Run the fund manager on macOS: launchd jobs from the crontab, agents from the units.

    python3 deploy/macos/launchd.py plan              # what would be installed
    python3 deploy/macos/launchd.py install           # write and load ~/Library/LaunchAgents
    python3 deploy/macos/launchd.py uninstall

The schedule is not restated here. Every active line of `deploy/cron.example` becomes a
launchd job with the same times, and every `deploy/fundmgr-*.service` becomes an agent
kept alive by launchd, so the Pi's table and the Mac's cannot drift apart. Lines with the
same command (the three backup times) share one job.

Why launchd and not cron on the Mac: the Mac sleeps outside its awake windows
(ai-studio docs/power.md). Cron skips a time it slept through; launchd runs a missed
calendar job once on wake. With windows set right nothing is missed, and a failed wake
costs a late run rather than none.

Every job runs under `studio power run --` (ADR 0010: workloads keep the Mac awake
through ai-studio, not with their own power code), so a `fund run` or `paper-track`
that starts near the end of a window is not put to sleep halfway through.

Stdlib only, Python 3.9+.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "com.fundmgr."
AGENTS = Path.home() / "Library" / "LaunchAgents"
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
TIMEZONE = "Europe/Stockholm"

RANGES = {"minute": (0, 59), "hour": (0, 23), "day": (1, 31), "month": (1, 12), "weekday": (0, 7)}
KEYS = {"minute": "Minute", "hour": "Hour", "day": "Day", "month": "Month", "weekday": "Weekday"}


@dataclass
class Job:
    label: str
    command: str
    intervals: list[dict] = field(default_factory=list)
    log: str | None = None


@dataclass
class Agent:
    label: str
    argv: list[str]
    env: dict[str, str]
    log: str


def expand(spec: str, name: str) -> list[int] | None:
    """Values a cron field allows, or None for every value."""
    lo, hi = RANGES[name]
    if spec == "*":
        return None
    values: set[int] = set()
    for part in spec.split(","):
        base, _, step = part.partition("/")
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            a, b = base.split("-")
            start, end = int(a), int(b)
        else:
            start = end = int(base)
        if not (lo <= start <= end <= hi):
            raise ValueError(f"{name} out of range: {part}")
        values.update(range(start, end + 1, int(step) if step else 1))
    if name == "weekday" and 7 in values:  # both mean Sunday; launchd takes 0
        values.discard(7)
        values.add(0)
    return sorted(values)


def intervals(fields: list[str]) -> list[dict]:
    names = ["minute", "hour", "day", "month", "weekday"]
    expanded = {n: expand(f, n) for n, f in zip(names, fields)}
    if expanded["day"] is not None and expanded["weekday"] is not None:
        # cron ORs these two; launchd ANDs every key in a dict. Not used in our table.
        raise ValueError("a line restricting both day-of-month and weekday is not supported")
    out: list[dict] = [{}]
    for n in names:
        if expanded[n] is None:
            continue
        out = [{**d, KEYS[n]: v} for d in out for v in expanded[n]]
    return out


CD = re.compile(r"^cd\s+(\S+)\s*&&\s*")
LOG = re.compile(r">>\s*(\S+?)\s+2>&1\s*$")


def parse_crontab(text: str) -> list[Job]:
    jobs: dict[str, Job] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or re.match(r"^[A-Z_]+=", line):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6 or parts[0].startswith("@"):
            raise ValueError(f"unsupported crontab line: {line}")
        command = CD.sub("", parts[5])  # launchd sets the working directory instead
        log_match = LOG.search(command)
        log = log_match.group(1) if log_match else None
        if command not in jobs:
            stem = Path(log).stem if log else re.sub(r"\W+", "-", command)[:40].strip("-")
            label = PREFIX + stem
            if any(j.label == label for j in jobs.values()):
                label += f"-{len(jobs)}"
            jobs[command] = Job(label=label, command=command, log=log)
        jobs[command].intervals += [i for i in intervals(parts[:5])
                                    if i not in jobs[command].intervals]
    return list(jobs.values())


def parse_unit(text: str, pi_root: str, root: Path) -> Agent:
    exec_start, env = "", {}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "ExecStart":
            exec_start = value
        elif key == "Environment":
            k, _, v = value.partition("=")
            env[k] = v.replace(pi_root, str(root))
    argv = [a.replace(pi_root, str(root)) for a in shlex.split(exec_start)]
    return Agent(label="", argv=argv, env=env, log="")


def pi_root_of(text: str) -> str:
    m = re.search(r"^WorkingDirectory=(\S+)", text, re.MULTILINE)
    if not m:
        raise ValueError("unit has no WorkingDirectory")
    return m.group(1)


def agents(units: Path, root: Path) -> list[Agent]:
    out = []
    for unit in sorted(units.glob("fundmgr-*.service")):
        text = unit.read_text()
        agent = parse_unit(text, pi_root_of(text), root)
        name = unit.stem.removeprefix("fundmgr-")
        agent.label, agent.log = PREFIX + name, f"data/logs/service-{name}.log"
        out.append(agent)
    return out


def env_for(root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    home = str(Path.home())
    return {"PATH": f"{root}/.venv/bin:{home}/.local/bin:{PATH}", "HOME": home, **(extra or {})}


def job_plist(job: Job, root: Path, studio: Path) -> dict:
    return {
        "Label": job.label,
        "ProgramArguments": [str(studio), "power", "run", "--", "/bin/bash", "-c", job.command],
        "WorkingDirectory": str(root),
        "StartCalendarInterval": job.intervals,
        "EnvironmentVariables": env_for(root),
        # The command appends to its own log; this catches failures before it can.
        "StandardOutPath": str(root / "data/logs/launchd.log"),
        "StandardErrorPath": str(root / "data/logs/launchd.log"),
        "ProcessType": "Background",
    }


def agent_plist(agent: Agent, root: Path) -> dict:
    return {
        "Label": agent.label,
        "ProgramArguments": agent.argv,
        "WorkingDirectory": str(root),
        "EnvironmentVariables": env_for(root, agent.env),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},  # systemd's Restart=on-failure
        "ThrottleInterval": 30,
        "StandardOutPath": str(root / agent.log),
        "StandardErrorPath": str(root / agent.log),
    }


def find_studio(explicit: str | None) -> Path | None:
    candidates = [explicit] if explicit else [
        os.environ.get("STUDIO_BIN"), str(ROOT.parent / "ai-studio" / ".venv" / "bin" / "studio"),
        str(Path.home() / "ai-studio" / ".venv" / "bin" / "studio")]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser().resolve()
    return None


def system_timezone() -> str | None:
    try:
        return os.readlink("/etc/localtime").split("zoneinfo/")[-1]
    except OSError:
        return os.environ.get("TZ")


def build(root: Path, crontab: Path, studio: Path) -> dict[str, dict]:
    plists = {j.label: job_plist(j, root, studio) for j in parse_crontab(crontab.read_text())}
    for a in agents(crontab.parent, root):
        plists[a.label] = agent_plist(a, root)
    return plists


def describe(plist: dict) -> str:
    if "StartCalendarInterval" in plist:
        times = plist["StartCalendarInterval"]
        return f"{len(times)} calendar entries: {plist['ProgramArguments'][-1]}"
    return "kept alive: " + " ".join(plist["ProgramArguments"])


def launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def cmd_plan(args, plists: dict[str, dict]) -> int:
    for label, plist in plists.items():
        print(f"{label}\n    {describe(plist)}")
    return 0


def cmd_install(args, plists: dict[str, dict]) -> int:
    root = Path(args.root)
    (root / "data" / "logs").mkdir(parents=True, exist_ok=True)
    AGENTS.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    for label, plist in plists.items():
        path = AGENTS / f"{label}.plist"
        with path.open("wb") as f:
            plistlib.dump(plist, f)
        if args.dry_run:
            print(f"wrote {path} (not loaded)")
            continue
        launchctl("bootout", f"{domain}/{label}")
        # Right after a bootout launchd can refuse the bootstrap ("5: Input/output error")
        # while the old job is still going away (as ai-studio's install-launchd.sh found).
        for attempt in range(5):
            r = launchctl("bootstrap", domain, str(path))
            if r.returncode == 0:
                break
            time.sleep(2 if attempt < 4 else 0)
        print(f"{'loaded' if r.returncode == 0 else 'FAILED'} {label}"
              + (f": {r.stderr.strip()}" if r.returncode else ""))
    for stale in sorted(AGENTS.glob(f"{PREFIX}*.plist")):
        if stale.stem not in plists:
            print(f"note: {stale} is not in the table any more; `uninstall` then `install` "
                  "removes it")
    return 0


def cmd_uninstall(args, plists: dict[str, dict]) -> int:
    domain = f"gui/{os.getuid()}"
    for path in sorted(AGENTS.glob(f"{PREFIX}*.plist")):
        launchctl("bootout", f"{domain}/{path.stem}")
        path.unlink()
        print(f"removed {path.stem}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--crontab", default=str(ROOT / "deploy" / "cron.example"))
    ap.add_argument("--studio", help="ai-studio's `studio` CLI (default: ../ai-studio/.venv/bin/studio)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(func=cmd_plan)
    ins = sub.add_parser("install")
    ins.add_argument("--dry-run", action="store_true", help="write the plists, load nothing")
    ins.set_defaults(func=cmd_install)
    sub.add_parser("uninstall").set_defaults(func=cmd_uninstall)
    args = ap.parse_args(argv)

    if args.cmd == "uninstall":
        return cmd_uninstall(args, {})
    studio = find_studio(args.studio)
    if studio is None:
        print("ai-studio's `studio` CLI not found; pass --studio /path/to/ai-studio/.venv/bin/studio",
              file=sys.stderr)
        return 1
    tz = system_timezone()
    if tz and tz != TIMEZONE:
        print(f"⚠ this Mac's time zone is {tz}; the table's times are {TIMEZONE} "
              f"(sudo systemsetup -settimezone {TIMEZONE})", file=sys.stderr)
    root = Path(args.root).resolve()
    args.root = str(root)
    return args.func(args, build(root, Path(args.crontab), studio))


if __name__ == "__main__":
    sys.exit(main())
