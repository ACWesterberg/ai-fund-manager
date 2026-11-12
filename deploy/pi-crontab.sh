#!/usr/bin/env bash
# pi-crontab.sh — hand the Pi's crontab over to the Mac move without touching anything
# that is not the fund's. DeepSwing (and anything else) shares this user's crontab, so
# `crontab -r` or installing a fresh table would silently stop it.
#
#   deploy/pi-crontab.sh strip      # cutover: remove the fund's jobs, keep every other line
#   deploy/pi-crontab.sh watchdog   # then: add deploy/cron.pi-watchdog's lines (idempotent)
#   deploy/pi-crontab.sh restore    # roll back: put the copy strip saved back
#
# A fund line is any line that mentions this checkout, plus the managed_cron.sh block.
# strip saves the whole table first, to ~/crontab.before-mac (never overwritten).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${FUND_DIR:-$SCRIPT_DIR/..}" && pwd)"
SAVED="${CRONTAB_BACKUP:-$HOME/crontab.before-mac}"
BEGIN="# >>> fundmgr managed (deploy/managed_cron.sh)"
END="# <<< fundmgr managed <<<"
WD_BEGIN="# >>> fundmgr watchdog (deploy/cron.pi-watchdog) >>>"
WD_END="# <<< fundmgr watchdog <<<"

current="$(crontab -l 2>/dev/null || true)"

case "${1:-}" in
strip)
    [ -e "$SAVED" ] || printf '%s\n' "$current" > "$SAVED"
    kept="$(printf '%s\n' "$current" | awk -v b="$BEGIN" -v e="$END" -v repo="$REPO_DIR" '
        index($0, b) == 1 {skip = 1; next}
        $0 == e {skip = 0; next}
        skip {next}
        index($0, repo) || /ai-fund-manager/ {
            if ($0 ~ /^[[:space:]]*#/) next      # a commented-out fund job: drop it too
            removed++; next
        }
        {print}
        END {print removed + 0 > "/dev/stderr"}' 2>"${TMPDIR:-/tmp}/pi-crontab.$$")"
    removed="$(cat "${TMPDIR:-/tmp}/pi-crontab.$$")"; rm -f "${TMPDIR:-/tmp}/pi-crontab.$$"
    printf '%s\n' "$kept" | crontab -
    echo "removed $removed fund job(s); kept $(printf '%s\n' "$kept" | grep -cvE '^[[:space:]]*(#|$)|^[A-Za-z_][A-Za-z0-9_]*=' || true) other job(s). Saved the old table: $SAVED"
    ;;
watchdog)
    rest="$(printf '%s\n' "$current" | awk -v b="$WD_BEGIN" -v e="$WD_END" '
        $0 == b {skip = 1; next} $0 == e {skip = 0; next} !skip')"
    block="$(sed "s#/home/alexander/Documents/ai-fund-manager#$REPO_DIR#g" "$REPO_DIR/deploy/cron.pi-watchdog")"
    printf '%s\n%s\n%s\n%s\n' "${rest%$'\n'}" "$WD_BEGIN" "$block" "$WD_END" | sed '/./,$!d' | crontab -
    echo "watchdog installed: $(printf '%s\n' "$block" | grep -c 'watchdog.sh') schedule line(s)"
    ;;
restore)
    [ -f "$SAVED" ] || { echo "no saved crontab at $SAVED" >&2; exit 1; }
    crontab "$SAVED" && echo "restored the crontab from $SAVED"
    ;;
*)
    echo "usage: $0 strip|watchdog|restore" >&2; exit 2 ;;
esac
