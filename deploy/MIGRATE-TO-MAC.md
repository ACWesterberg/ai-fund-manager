# Moving the fund manager from the Pi to the Mac Studio

The Mac becomes the fund's host. ai-studio runs the machine (sleep and wake, the gateway,
the dashboard); this repo brings its own launchd jobs (ai-studio ADR 0009, 0010). The
Pi stays on as the **watchdog**.

## What has to come across, and why all at once

Everything git does not hold is state, and most of it exists only on the Pi:

| State | Where | Replaceable? |
| --- | --- | --- |
| The five fund ledgers, NAV, recommendations, outcomes, learnings | `data/fund*.db` | **No** |
| Live and paper sleeves (real broker mirrors among them) | `data/paper/*.db` | **No**, and missing from the old daily backup |
| Active guidance, inactive candidates, search checkpoints, batch ids, request cache | `config/compiled/` (not in git) | **No** (paid for) |
| Shadow comparisons (immutable, scored later) | `data/shadow/` | **No** |
| What-If results, ticker aliases, reports, logs | `data/whatif/`, `data/ticker_aliases.json`, `data/reports*/`, `data/logs/` | Mostly no |
| API keys, Telegram, dashboard password | `.env` | Re-enterable |
| FinanceData cache (prices, fundamentals, news) | `~/.financedata/` | Prices yes; aged-off news no |

The funds cannot be split across machines: each fund's optimizer pools training
examples from the other four databases (`optimizer.pool_configs`), and one Telegram bot
and one dashboard serve all five. So the real book moves in the same cutover as the
sims, and the watchdog has to exist first (ADR 0009). It does: `deploy/watchdog.sh`.

`deploy/migrate.py` moves all of the above in one archive. It finds the state by asking
git what it does not track rather than from a list, snapshots every SQLite file through
the online backup API, and writes a manifest with each file's hash and each database's
row counts and integrity check. `restore` refuses to finish unless every file matches the
manifest, and never overwrites a differing file without `--force`.

## 0. The Mac, once

ai-studio must already be running there (its `docs/day-one.md`), then:

1. **Time zone** Europe/Stockholm, since every job time is local:
   `sudo systemsetup -settimezone Europe/Stockholm`
2. **Awake windows** in ai-studio's `config/ai-studio.yaml`, with `power.enabled: true`
   (ai-studio `docs/power.md`). The fund needs:
   - **weekdays 07:00-23:59**: news, stops on both markets, Monday runs, `paper-track` at
     22:15, and room after the US close (22:00) for reading and analysing reports that
     come out then. A window cannot cross midnight, so 23:59 is the latest end.
   - Sunday 01:55-04:15: the five `fund optimize` checks
   ```yaml
   power:
     enabled: true
     windows:
       - {name: trading day, days: [mon, tue, wed, thu, fri], start: "07:00", end: "23:59"}
       - {name: weekly night job, days: [sun], start: "01:55", end: "04:15"}
   ```
   A job started inside a window always finishes, because it runs under
   `studio power run`. The window matters for jobs that start late, such as a US report
   job at 22:45.

   Check that the finance window under `gpu.reserve` is on (it was turned off until the
   fund moved).
3. **Log in automatically** at boot (System Settings → Users & Groups). The jobs are
   LaunchAgents, like ai-studio's own, and only start in a logged-in session.
4. **Tools**: `brew install tesseract tesseract-lang rclone`, and uv
   (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
5. **rclone**: copy `~/.config/rclone/rclone.conf` from the Pi, then `rclone lsd gdrive:`.
6. **GitHub SSH key** (`deploy/SETUP.md` §2), then clone both repos side by side and install:
   ```bash
   cd ~ && git clone git@github.com:acwesterberg/ai-fund-manager.git
   git clone git@github.com:acwesterberg/FinanceData.git
   cd ai-fund-manager && git checkout deploy
   uv venv .venv && uv pip install -e ../FinanceData && uv pip install -e .
   ```
   Put it next to `ai-studio` (or pass `--studio` below).

## 1. Rehearse (the Pi keeps running)

On the Pi:
```bash
cd ~/Documents/ai-fund-manager && python3 deploy/migrate.py export --live
scp ~/fund-migration-*.tgz mac-studio:
```
On the Mac:
```bash
cd ~/ai-fund-manager && python3 deploy/migrate.py restore ~/fund-migration-*.tgz
.venv/bin/fund status
FUND_CONFIG=config/config_global.yaml .venv/bin/fund status
.venv/bin/fund run --dry-run --skip-news
.venv/bin/uvicorn fundmgr.web.app:app --port 8010    # look, then Ctrl-C
python3 deploy/macos/launchd.py plan
```
During the rehearsal, **do not start the bot or install the jobs on the Mac.** Two bots
on one Telegram token fight over updates, and two sets of jobs would fill the
simulations twice.

## 2. Cut over

Choose a Saturday: no market, no scheduled job, and Sunday's optimize run will be the
first job on the Mac.

On the Pi, stop everything that writes:
```bash
crontab -l > ~/crontab.before-mac && crontab -r
pgrep -af fund        # wait until nothing but the services is left
sudo systemctl disable --now fundmgr-bot fundmgr-web fundmgr-global-web
cd ~/Documents/ai-fund-manager && python3 deploy/migrate.py export
scp ~/fund-migration-*.tgz mac-studio:
```
`export` refuses while a fund process or a crontab line that mentions the repo is still
there. That is deliberate: anything written after the snapshot would stay on the Pi.

On the Mac:
```bash
cd ~/ai-fund-manager
python3 deploy/migrate.py restore ~/fund-migration-<final>.tgz --force   # --force only over a rehearsal
python3 deploy/macos/launchd.py install
python3 deploy/migrate.py verify
```
Then check that `/status` answers in Telegram, the dashboard loads at `http://mac-studio:8010`
(sims under `/sim*`), and `launchctl list | grep fundmgr` lists the jobs.
Delete the archives once restored: they contain `.env`.

On the Pi, switch to the watchdog role and park the old data where nothing will use it
by accident:
```bash
cd ~/Documents/ai-fund-manager
mv data data.moved-to-mac-$(date +%Y%m%d) && mkdir -p data/logs
crontab deploy/cron.pi-watchdog
MAC_FUND_URL=http://mac-studio:8010/ deploy/watchdog.sh    # prints nothing when the Mac is up
```
Keep `data.moved-to-mac-*` for a few weeks. It is the fallback if the Mac copy has a problem.

**Deploys:** in GitHub → Settings → Secrets, point `PI_TAILSCALE_IP`, `PI_SSH_USER`
and `PI_SSH_KEY` at the Mac, and add `FUND_DIR=/Users/<you>/ai-fund-manager`. Turn on
Remote Login on the Mac (System Settings → General → Sharing). `deploy.sh` detects macOS
and restarts the launchd agents. Or use the 5-minute poll: add
`*/5 * * * * cd ~/ai-fund-manager && FUND_DIR=$PWD deploy/poll-deploy.sh` to the Mac's crontab,
or ask for a launchd job for it.

## 3. What changes in day-to-day running

- **The Mac sleeps outside its windows** (weekdays after midnight until 06:50, and
  Saturday). The bot and the dashboards are unreachable then. Telegram holds messages for
  24 hours, so a fill screenshot sent at 01:00 is read on the morning wake.
- **A job time the Mac slept through runs once on wake** (launchd), not never (cron).
  The 03:00 weekday backup therefore runs at the 06:50 wake, and Saturday's runs on
  Sunday night. The Monday 11:00 and 18:00 snapshots fall inside the window.
- **Every job runs under `studio power run`**, so a long `paper-track` or report analysis
  finishes even if it runs past midnight.
- **Backups** stay this repo's own (`deploy/backup.sh` → Google Drive). It now takes the
  same export as the migration, so live sleeves, compiled guidance and shadow comparisons
  are included. ai-studio's `pg_dumpall` does not cover SQLite.
- **The watchdog** alerts once if the Mac does not answer for two checks in a row between
  08:45 and 22:15 (inside the window, covering every market-hours job), and again when it recovers. It does not yet run `check-stops`
  itself.

## Rolling back

Stop the Mac (`python3 deploy/macos/launchd.py uninstall`). Then take a final
`python3 deploy/migrate.py export` on the Mac, restore it on the Pi (`--force`), put the
old crontab back with `crontab ~/crontab.before-mac`, and run
`sudo systemctl enable --now fundmgr-bot fundmgr-web fundmgr-global-web`. The same
tool works in both directions.
