#!/usr/bin/env bash
# managed_cron.sh — keep the deploy-managed cron jobs in the Pi's crontab.
# Called by deploy.sh. Everything else in the crontab is hand-maintained from
# deploy/cron.example and is never touched: only the lines between the two
# markers below are rewritten, so re-running is a no-op and editing a job here
# and deploying is how it changes. Advisory — a failure here never fails a deploy.
set -uo pipefail

REPO_DIR="$(cd "${1:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
BEGIN="# >>> fundmgr managed (deploy/managed_cron.sh) — edit there, not here >>>"
END="# <<< fundmgr managed <<<"

block() {
    cat <<CRON
$BEGIN
# Broker accounts (/accounts): read every Montrose-linked account each 30 min in
# trading hours and book what changed. Read-only; an LLM call only for a newly
# bought name; Telegram only on a change.
*/30 9-22 * * 1-5 cd $REPO_DIR && .venv/bin/fund account-sync --all >> data/logs/accounts.log 2>&1
# Refresh every account holding's hold/add/trim/exit view, Saturday morning.
41 8 * * 6 cd $REPO_DIR && .venv/bin/fund account-plan >> data/logs/accounts.log 2>&1
$END
CRON
}

current="$(crontab -l 2>/dev/null || true)"
# Drop any previous managed block, keep every other line exactly as it was.
rest="$(printf '%s\n' "$current" | awk -v b="$BEGIN" -v e="$END" '
    $0 == b {skip = 1; next}
    $0 == e {skip = 0; next}
    !skip')"
# A job already added by hand outside the block would run twice — say so, and
# leave it to the operator rather than guess which copy to keep.
if printf '%s\n' "$rest" | grep -v '^[[:space:]]*#' | grep -qE 'fund account-(sync --all|plan)'; then
    echo "managed_cron: account jobs already in the crontab outside the managed block — not adding them again"
    exit 0
fi
wanted="$(printf '%s\n%s\n' "${rest%$'\n'}" "$(block)" | sed '/./,$!d')"
if [ "$wanted" = "$current" ]; then
    echo "managed_cron: crontab already up to date"
    exit 0
fi
printf '%s\n' "$wanted" | crontab - && echo "managed_cron: crontab updated"
