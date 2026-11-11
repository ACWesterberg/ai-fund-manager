#!/usr/bin/env bash
# watchdog.sh — runs on the Pi once the fund lives on the Mac Studio (ADR 0009 in
# ai-studio). The Mac sleeps outside its awake windows and wakes itself; a wake that
# fails (power cut, stuck update) cannot be noticed from the Mac, and a missed
# check-stops on the real book is the job that costs money. So the Pi asks.
#
# Any HTTP answer from the fund dashboard counts as up: without credentials it answers
# with a redirect to /login (or 401), which proves the Mac is awake and serving. No answer for FAILS_BEFORE_ALERT
# checks in a row sends one Telegram alert; recovery sends another. Nothing repeats in
# between, so a long outage is one message, not one every 15 minutes.
#
# It does not run check-stops itself: the real book's stops live in the Mac's fund.db.
# Until that fallback exists, the alert means "check your positions in Montrose".
#
# Schedule: deploy/cron.pi-watchdog (only inside the Mac's awake window; asleep is not down).
#
# Env: MAC_FUND_URL (default http://mac-studio:8010/), FAILS_BEFORE_ALERT (default 2),
#      TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (read from .env when not set).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${FUND_DIR:-$SCRIPT_DIR/..}"
URL="${MAC_FUND_URL:-http://mac-studio:8010/}"
THRESHOLD="${FAILS_BEFORE_ALERT:-2}"
STATE="$REPO_DIR/data/watchdog.state"
mkdir -p "$REPO_DIR/data"

from_env() {
    [ -f "$REPO_DIR/.env" ] || return 0
    sed -n "s/^$1=//p" "$REPO_DIR/.env" | tail -n1 | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}
TOKEN="${TELEGRAM_BOT_TOKEN:-$(from_env TELEGRAM_BOT_TOKEN)}"
CHAT="${TELEGRAM_CHAT_ID:-$(from_env TELEGRAM_CHAT_ID)}"

say() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    if [ -n "$TOKEN" ] && [ -n "$CHAT" ]; then
        curl -fsS -m 20 "https://api.telegram.org/bot$TOKEN/sendMessage" \
            --data-urlencode "chat_id=$CHAT" --data-urlencode "text=$*" >/dev/null \
            || echo "  (Telegram send failed)"
    fi
}

fails=0 alerted=0
[ -f "$STATE" ] && read -r fails alerted < "$STATE" || true

code="$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$URL" || true)"
if [ -n "$code" ] && [ "$code" != "000" ]; then
    [ "$alerted" = "1" ] && say "✅ Mac Studio answers again ($URL, HTTP $code). Fund jobs resume on their schedule; a missed check-stops runs once on wake."
    echo "0 0" > "$STATE"
    exit 0
fi

fails=$((fails + 1))
if [ "$fails" -ge "$THRESHOLD" ] && [ "$alerted" != "1" ]; then
    say "🚨 Mac Studio not answering ($URL) for $fails checks. Stops on the real book are NOT being checked. Look at your positions in Montrose, and wake or power-cycle the Mac."
    alerted=1
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] no answer from $URL ($fails/$THRESHOLD)"
fi
echo "$fails $alerted" > "$STATE"
