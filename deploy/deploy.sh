#!/usr/bin/env bash
# deploy.sh — runs ON the Raspberry Pi to pull latest code and restart services.
# Called by GitHub Actions (via SSH) or by the polling script.
set -euo pipefail

# Derive repo root from this script's location so it works regardless of username or path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${FUND_DIR:-$SCRIPT_DIR/..}"
BRANCH="${DEPLOY_BRANCH:-deploy}"
LOG_FILE="$REPO_DIR/data/logs/deploy.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# The Pi once sat on a FinanceData feature branch for weeks without anyone
# noticing: nothing checked, and a stale data layer produces plausible wrong
# numbers rather than an error. This says so on every deploy, and repoints
# itself when — and only when — doing so cannot lose work.
check_financedata_branch() {
    local dir="$FINANCEDATA_DIR" want="$FINANCEDATA_BRANCH" have
    have=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
    [ "$have" = "$want" ] && return 0

    log "  ⚠ FinanceData is on '$have', expected '$want'"
    git -C "$dir" fetch origin "$want" --quiet 2>/dev/null || {
        log "  ⚠ could not fetch origin/$want — leaving the checkout alone"
        return 0
    }
    # Only switch when this branch is already contained in the target and the
    # tree is clean: then the checkout holds nothing that isn't on $want, and
    # moving it discards nothing. Anything else is the operator's call — an
    # unmerged branch may be deliberate, and a deploy script must not decide
    # that for them.
    if [ -n "$(git -C "$dir" status --porcelain)" ]; then
        log "  ⚠ working tree is dirty — NOT switching. Fix by hand:"
        log "      git -C $dir status"
        return 0
    fi
    if git -C "$dir" merge-base --is-ancestor HEAD "origin/$want" 2>/dev/null; then
        git -C "$dir" checkout "$want" --quiet && git -C "$dir" pull --ff-only --quiet || true
        log "  ✓ repointed FinanceData to $want (nothing was lost — '$have' is already merged)"
    else
        log "  ⚠ '$have' has commits not on $want — NOT switching, to avoid discarding them."
        log "      Merge or discard, then: git -C $dir checkout $want"
    fi
}

mkdir -p "$(dirname "$LOG_FILE")"
# Serialize webhook, polling and SSH deployments.
exec 9>"$REPO_DIR/data/deploy.lock"
flock -n 9 || { log "Another deployment is running"; exit 0; }
SUCCESS_FILE="$REPO_DIR/data/deployed-revision"
log "=== Deploy started (branch: $BRANCH) ==="

cd "$REPO_DIR"

# Wait before changing the source or dependencies used by an active run.
log "Checking for active fund run…"
WAIT=0
while pgrep -f "fund run" > /dev/null 2>&1; do
    if [ $WAIT -eq 0 ]; then log "  Fund run in progress — waiting for it to finish…"; fi
    WAIT=$((WAIT + 5))
    if [ $WAIT -gt 1800 ]; then
        log "  ✗ Fund run still active — aborting deployment; retry later"
        exit 1
    fi
    sleep 5
done
[ $WAIT -gt 0 ] && log "  Fund run finished after ${WAIT}s — proceeding with restart"

# Ensure we're on the right branch
git fetch origin "$BRANCH" --quiet
TARGET=$(git rev-parse "origin/$BRANCH")
if [ -n "${EXPECTED_REVISION:-}" ] && [ "$TARGET" != "$EXPECTED_REVISION" ]; then
    log "Branch advanced beyond the tested revision — aborting; run checks again"
    exit 1
fi
BEFORE=$(git rev-parse HEAD)
git reset --hard "origin/$BRANCH" --quiet
AFTER=$(git rev-parse HEAD)

if [ -f "$SUCCESS_FILE" ] && [ "$(cat "$SUCCESS_FILE")" = "$AFTER" ]; then
    log "Already up to date ($AFTER). Nothing to do."
    exit 0
fi

log "Updated $BEFORE → $AFTER"
git log --oneline "$BEFORE..$AFTER" | while read -r line; do log "  $line"; done

# Install / update Python dependencies
log "Updating dependencies…"
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
# Shared financedata package: pull its latest source and (re)install it first.
# Resolve both editable projects explicitly, honoring FINANCEDATA_DIR rather
# than the default sibling path in tool.uv.sources. Git-pull is non-fatal so a
# FinanceData hiccup cannot block the deploy —
# but it is never silent, because a data layer running unnoticed off the wrong
# branch shows up as wrong numbers, not as a failed deploy.
FINANCEDATA_DIR="${FINANCEDATA_DIR:-$REPO_DIR/../FinanceData}"
FINANCEDATA_BRANCH="${FINANCEDATA_BRANCH:-main}"
if [ -d "$FINANCEDATA_DIR" ]; then
    if [ -d "$FINANCEDATA_DIR/.git" ]; then
        log "Updating FinanceData ($FINANCEDATA_DIR)…"
        check_financedata_branch
        if git -C "$FINANCEDATA_DIR" pull --ff-only --quiet; then
            log "  FinanceData → $(git -C "$FINANCEDATA_DIR" rev-parse --short HEAD) on $(git -C "$FINANCEDATA_DIR" rev-parse --abbrev-ref HEAD)"
        else
            log "  ⚠ FinanceData git pull failed — installing current checkout $(git -C "$FINANCEDATA_DIR" rev-parse --short HEAD)"
        fi
    else
        log "  ⚠ $FINANCEDATA_DIR is not a git checkout — installing it as-is, with no way to tell how old it is"
    fi
else
    log "  ✗ $FINANCEDATA_DIR not found — cannot deploy"
    exit 1
fi
"$UV" pip install --no-sources -e "$FINANCEDATA_DIR" -e . --quiet

# Refresh universe CSVs from FinanceData when the sibling repo is present.
if [ -x "$UV" ] && [ -f "$REPO_DIR/scripts/sync_universe_from_financedata.py" ]; then
    if FINANCEDATA_DIR="$FINANCEDATA_DIR" "$REPO_DIR/.venv/bin/python" "$REPO_DIR/scripts/sync_universe_from_financedata.py" 2>&1 | tee -a "$LOG_FILE"; then
        log "Universe synced from FinanceData"
    else
        log "  ⚠ Universe sync skipped or failed — using committed CSVs"
    fi
fi

log "Restarting services…"
sudo systemctl restart fundmgr-bot fundmgr-web fundmgr-global-web

# Verify services came back up
sleep 2
for svc in fundmgr-bot fundmgr-web fundmgr-global-web; do
    if systemctl is-active --quiet "$svc"; then
        log "  ✓ $svc is running"
    else
        log "  ✗ $svc FAILED to start"
        systemctl status "$svc" --no-pager -l >> "$LOG_FILE" 2>&1
        exit 1
    fi
done

printf '%s\n' "$AFTER" > "$SUCCESS_FILE.tmp"
mv "$SUCCESS_FILE.tmp" "$SUCCESS_FILE"
log "=== Deploy complete ==="
