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

mkdir -p "$(dirname "$LOG_FILE")"
log "=== Deploy started (branch: $BRANCH) ==="

cd "$REPO_DIR"

# Ensure we're on the right branch
git fetch origin "$BRANCH" --quiet
BEFORE=$(git rev-parse HEAD)
git reset --hard "origin/$BRANCH" --quiet
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
    log "Already up to date ($AFTER). Nothing to do."
    exit 0
fi

log "Updated $BEFORE → $AFTER"
git log --oneline "$BEFORE..$AFTER" | while read -r line; do log "  $line"; done

# Install / update Python dependencies
log "Updating dependencies…"
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
"$UV" pip install -e . --quiet

# Restart services (requires sudoers entry — see SETUP.md)
log "Restarting services…"
sudo systemctl restart fundmgr-bot fundmgr-web

# Verify services came back up
sleep 2
for svc in fundmgr-bot fundmgr-web; do
    if systemctl is-active --quiet "$svc"; then
        log "  ✓ $svc is running"
    else
        log "  ✗ $svc FAILED to start"
        systemctl status "$svc" --no-pager -l >> "$LOG_FILE" 2>&1
        exit 1
    fi
done

log "=== Deploy complete ==="
