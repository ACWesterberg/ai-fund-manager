# Backups — SQLite → Google Drive (rclone)

Backs up the fund databases (real + both sims) and configs to Google Drive on a
schedule, using SQLite's consistent online backup. Code/configs are already in
git; the **databases are the irreplaceable part** (real-fund ledger, NAV
history, recommendations), so this protects them against another card failure.

## 1. Install rclone (once, on the Pi)

```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone version
```

## 2. Configure a Google Drive remote (once)

The Pi is headless, so authorise on a machine that has a browser:

```bash
rclone config
#  n) New remote
#  name> gdrive
#  storage> drive               (Google Drive)
#  client_id / client_secret>   (leave blank, or add your own for higher limits)
#  scope> 1                     (full access) or 3 (drive.file — only files rclone creates)
#  Edit advanced config? n
#  Use auto config? n           ← headless: say NO
#     → it prints:  rclone authorize "drive"
#  Run THAT command on your laptop (has a browser), log in, copy the token,
#  paste it back into the Pi prompt.
#  Configure as team drive? n
#  y) Yes this is OK
```

Verify:
```bash
rclone lsd gdrive:          # lists your Drive folders
```

The default remote path is `gdrive:fund-manager-backups` (rclone creates it).
Override with the `RCLONE_REMOTE` env var if you want a different name/folder.

## 3. Test a backup manually

```bash
cd ~/Documents/ai-fund-manager
deploy/backup.sh
rclone ls gdrive:fund-manager-backups     # should show a fund-backup-*.tgz
```

## 4. Schedule it (crontab -e)

```cron
# Daily DB backup — 03:00 CET
0 3 * * * cd /home/alexander/Documents/ai-fund-manager && deploy/backup.sh >> data/logs/backup.log 2>&1

# Extra snapshot right after the Monday runs (captures the post-decision state)
0 11 * * 1 cd /home/alexander/Documents/ai-fund-manager && deploy/backup.sh >> data/logs/backup.log 2>&1
0 18 * * 1 cd /home/alexander/Documents/ai-fund-manager && deploy/backup.sh >> data/logs/backup.log 2>&1
```

Retention defaults to 14 days (local + remote). Override with `RETENTION_DAYS`.

## 5. Restore (after a card failure / new SD)

```bash
cd ~/Documents/ai-fund-manager
mkdir -p data restore
# list backups, newest last:
rclone lsf gdrive:fund-manager-backups | sort
# pull the one you want:
rclone copy gdrive:fund-manager-backups/fund-backup-YYYYMMDD-HHMMSS.tgz restore/
tar -xzf restore/fund-backup-*.tgz -C restore/
# put the DBs back:
cp restore/*.db data/
# (configs are in git; only copy restore/config/* if you changed something locally)
```
Then restart services: `sudo systemctl restart fundmgr-bot fundmgr-web fundmgr-global-web`.

## Notes
- `.env` (API keys, Telegram token) is **not** backed up by default — it holds
  secrets. Keep those in a password manager, or set `BACKUP_ENV=1` only if the
  Drive folder is adequately private.
- The financedata cache (`~/.financedata/cache.db`) is intentionally **not**
  backed up — it's regenerated from the data APIs.
- SQLite `.backup` is consistent even mid-run, so it's safe to schedule anytime.
