from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pi(tmp_path):
    """A fake curl: the Mac answers with whatever is in mac-code; Telegram sends are logged."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "curl").write_text(
        "#!/bin/sh\n"
        'case "$*" in *api.telegram.org*) echo "$*" >> "$FUND_DIR/sent"; exit 0;; esac\n'
        'cat "$FUND_DIR/mac-code" 2>/dev/null || printf 000\n')
    (binaries / "curl").chmod(0o755)
    (tmp_path / ".env").write_text('TELEGRAM_BOT_TOKEN="123:abc"\nTELEGRAM_CHAT_ID=42\n')
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "FUND_DIR": str(tmp_path)}
    env.pop("TELEGRAM_BOT_TOKEN", None)
    env.pop("TELEGRAM_CHAT_ID", None)

    def check(code: str | None) -> list[str]:
        mac = tmp_path / "mac-code"
        if code is None:
            mac.unlink(missing_ok=True)
        else:
            mac.write_text(code)
        subprocess.run(["bash", str(ROOT / "deploy" / "watchdog.sh")], env=env, check=True,
                       capture_output=True)
        sent = tmp_path / "sent"
        return sent.read_text().splitlines() if sent.exists() else []
    return check


def test_one_alert_per_outage_and_one_on_recovery(pi):
    assert pi("401") == []          # auth-protected dashboard answering = up
    assert pi(None) == []           # one miss is not an outage
    sent = pi(None)
    assert len(sent) == 1 and "not answering" in sent[0] and "bot123:abc" in sent[0]
    assert "chat_id=42" in sent[0]
    assert len(pi(None)) == 1       # still down: no repeat
    sent = pi("200")
    assert len(sent) == 2 and "answers again" in sent[1]
    assert len(pi("401")) == 2      # up and already said so


def test_a_blip_that_recovers_says_nothing(pi):
    assert pi(None) == []
    assert pi("401") == []
    assert pi(None) == []
