#!/usr/bin/env bash
# poll-deploy.sh — runs as a cronjob on the Pi.
# Checks for new commits on the deploy branch every N minutes.
# No inbound networking required — Pi initiates outbound git fetch.
#
# Add to crontab (crontab -e):
#   */5 * * * * /home/pi/ai-fund-manager/deploy/poll-deploy.sh >> /home/pi/ai-fund-manager/data/logs/poll.log 2>&1
set -euo pipefail

REPO_DIR="${FUND_DIR:-/home/pi/ai-fund-manager}"
BRANCH="${DEPLOY_BRANCH:-deploy}"

cd "$REPO_DIR"

git fetch origin "$BRANCH" --quiet 2>&1 || exit 0

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] New commit detected ($LOCAL → $REMOTE), deploying…"
    DEPLOY_BRANCH="$BRANCH" bash "$REPO_DIR/deploy/deploy.sh"
fi
