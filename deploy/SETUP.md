# Raspberry Pi Setup Guide

This guide walks through setting up the AI Fund Manager on a Raspberry Pi 5 from scratch, including automatic deploys triggered by a `git push` to the `deploy` branch.

---

## Prerequisites

- Raspberry Pi 5 running Raspberry Pi OS Lite (64-bit, Bookworm)
- SSH access to the Pi on your local network
- A GitHub account with this repository
- API keys: OpenAI, Telegram bot token + chat ID

---

## 1. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl build-essential python3-dev
sudo apt install -y tesseract-ocr tesseract-ocr-swe tesseract-ocr-eng
```

Install `uv` (Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or re-login
```

---

## 2. Clone the repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/ai-fund-manager.git
cd ai-fund-manager
```

Create and activate the virtual environment, then install dependencies:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

---

## 3. Configure environment variables

```bash
cp .env.example .env
nano .env
```

Fill in at minimum:

```
OPENAI_API_KEY=sk-...           # GPT-5.5 simulation fund
ANTHROPIC_API_KEY=sk-ant-...    # Claude Opus simulation fund
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Optional (only needed if using the webhook deploy method):

```
DEPLOY_WEBHOOK_SECRET=your-random-secret
DEPLOY_BRANCH=deploy
```

---

## 4. Initialise the database

```bash
~/.venv/bin/fund init
```

This creates `data/fund.db` and seeds the cash balance (50 000 SEK).

---

## 5. Install systemd services

Copy the service files and enable them:

```bash
sudo cp deploy/fundmgr-bot.service /etc/systemd/system/
sudo cp deploy/fundmgr-web.service /etc/systemd/system/
```

Edit each file if your username is not `pi` or if you cloned the repo to a different path:

```bash
sudo nano /etc/systemd/system/fundmgr-bot.service
sudo nano /etc/systemd/system/fundmgr-web.service
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable fundmgr-bot fundmgr-web
sudo systemctl start  fundmgr-bot fundmgr-web
```

Check they came up:

```bash
systemctl status fundmgr-bot fundmgr-web
```

---

## 6. Sudoers entry (required for deploy script)

The deploy script restarts the services without a password prompt. Grant that permission:

```bash
sudo visudo -f /etc/sudoers.d/fundmgr
```

Add this line (replace `pi` with your username if different):

```
pi ALL=(ALL) NOPASSWD: /bin/systemctl restart fundmgr-bot, /bin/systemctl restart fundmgr-web
```

Save and exit. Verify with:

```bash
sudo systemctl restart fundmgr-bot fundmgr-web
```

---

## 7. Set timezone + cron jobs

Set the Pi timezone so cron times track DST automatically:

```bash
sudo timedatectl set-timezone Europe/Stockholm
date   # should show CET or CEST
```

Then install the cron jobs:

```bash
crontab -e
```

Paste the contents of `deploy/cron.example` — see that file for the full schedule with comments. Summary:

| Fund | Job | Time (CET) | Reason |
|------|-----|------------|--------|
| Nordic | Weekly run | Mon 09:30 | 30 min after OMX opens |
| Nordic | News checks | Weekdays 09/11/14/16 | Full trading day coverage |
| Nordic | Stop-loss | Weekdays 17:45 | 15 min after OMX closes |
| Global sim | Weekly run | Mon 16:00 | NYSE open 30 min + EU still live |
| Global sim | Stop-loss | Weekdays 22:15 | 15 min after NYSE closes |

---

## 8. Auto-deploy — choose one method

### Method A: GitHub Actions + Tailscale (recommended)

No inbound port forwarding needed. GitHub Actions SSHes into the Pi over Tailscale.

#### 8a. Install Tailscale on the Pi

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Note the Tailscale IP shown (`100.x.x.x`). You'll add it as a GitHub secret.

#### 8b. Generate a dedicated SSH key pair

On the Pi:

```bash
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_deploy -N ""
cat ~/.ssh/github_deploy.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

The private key goes into GitHub secrets — copy it:

```bash
cat ~/.ssh/github_deploy
```

#### 8c. Create a Tailscale OAuth client

1. Go to [tailscale.com/admin/settings/oauth](https://login.tailscale.com/admin/settings/oauth)
2. Create a new client with the `auth_keys` scope
3. Add tag `tag:ci` under **Devices → Tags** in Tailscale admin first
4. Note the client ID and secret

#### 8d. Add GitHub repository secrets

In your repo → **Settings → Secrets and variables → Actions**, add:

| Secret name                  | Value                                      |
|------------------------------|--------------------------------------------|
| `PI_TAILSCALE_IP`            | `100.x.x.x` (your Pi's Tailscale IP)      |
| `PI_SSH_USER`                | `pi` (or your username)                    |
| `PI_SSH_KEY`                 | Contents of `~/.ssh/github_deploy`         |
| `TAILSCALE_OAUTH_CLIENT_ID`  | From step 8c                               |
| `TAILSCALE_OAUTH_CLIENT_SECRET` | From step 8c                            |
| `TELEGRAM_BOT_TOKEN`         | Your Telegram bot token                    |
| `TELEGRAM_CHAT_ID`           | Your Telegram chat ID                      |

#### 8e. Push to deploy

```bash
git checkout -b deploy
git push origin deploy
```

Every subsequent push to the `deploy` branch triggers the Actions workflow, which SSHes into the Pi and runs `deploy/deploy.sh`. You'll get a Telegram notification on success or failure.

You can also trigger a deploy manually from **Actions → Deploy to Raspberry Pi → Run workflow**.

---

### Method B: Polling (no Tailscale, no port forwarding)

The Pi polls GitHub for new commits every 5 minutes. Simpler but has up to a 5-minute delay.

```bash
crontab -e
```

Add:

```cron
*/5 * * * * /home/pi/ai-fund-manager/deploy/poll-deploy.sh >> /home/pi/ai-fund-manager/data/logs/poll.log 2>&1
```

No GitHub secrets or Tailscale required. The Pi needs outbound internet access to GitHub (standard).

---

### Method C: GitHub Webhook (direct HTTP, needs public URL)

GitHub POSTs to your Pi's FastAPI `/deploy` endpoint on every push.

Requires the Pi's port 8000 to be reachable from the internet — either via port forwarding on your router or a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/).

1. Add `DEPLOY_WEBHOOK_SECRET` (any random string) to `.env` on the Pi and restart `fundmgr-web`
2. In GitHub: **Settings → Webhooks → Add webhook**
   - Payload URL: `http://YOUR_PI_IP:8000/deploy`
   - Content type: `application/json`
   - Secret: the same random string
   - Event: **Just the push event**
3. Change `DEPLOY_BRANCH` in `.env` if you want a branch other than `deploy`

---

## 9. Verify the full setup

```bash
# Web dashboard
curl http://localhost:8000/

# Telegram bot — send /status from your Telegram app

# Manual deploy test
DEPLOY_BRANCH=deploy bash ~/ai-fund-manager/deploy/deploy.sh
```

---

---

## 10. Custom domain (Namecheap + Cloudflare Tunnel)

This lets you access the dashboard at `https://fund.yourdomain.com` from anywhere — no open ports, free SSL, works even behind a home router.

### 10a. Move DNS to Cloudflare (Namecheap side)

1. Log into [namecheap.com](https://www.namecheap.com) → **Domain List** → click **Manage** next to your domain
2. Under **Nameservers**, select **Custom DNS**
3. Enter Cloudflare's nameservers (you'll get these in step 10b):
   ```
   erin.ns.cloudflare.com
   josh.ns.cloudflare.com
   ```
   *(The exact names vary per account — copy them from Cloudflare)*
4. Click the green tick to save. DNS propagation takes up to 24h but usually under 30 minutes.

### 10b. Add your site to Cloudflare

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) → **Add a site**
2. Enter your domain name → choose the **Free plan**
3. Cloudflare scans your existing DNS records — keep any you need (MX records for email, etc.)
4. Copy the two nameservers shown and enter them in Namecheap (step 10a)
5. Click **Done, check nameservers** — Cloudflare will email you when it's active

### 10c. Create the tunnel (Cloudflare side)

1. In Cloudflare dashboard → **Zero Trust** → **Networks** → **Tunnels**
2. Click **Create a tunnel** → name it `fund-manager` → **Save tunnel**
3. Copy the tunnel token shown (a long string starting with `eyJ…`) — you'll need it on the Pi
4. Under **Public Hostname**, click **Add a public hostname**:
   - Subdomain: `fund` (or whatever you want — e.g. `dashboard`)
   - Domain: your domain
   - Service Type: `HTTP`, URL: `localhost:8000`
5. Click **Save hostname**

Your dashboard will be at `https://fund.yourdomain.com` once the tunnel is running.

### 10d. Install and configure cloudflared on the Pi

```bash
# Install cloudflared
curl -L https://pkg.cloudflare.com/cloudflare-main.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloudflare-main.gpg
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared bookworm main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
```

Install as a system service using the token from step 10c:

```bash
sudo cloudflared service install eyJ...YOUR_TOKEN_HERE...
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

Verify it's running:

```bash
systemctl status cloudflared
```

That's it — no port forwarding, no router config needed. Cloudflare handles HTTPS and the SSL certificate automatically.

### 10e. Optional: restrict access to yourself only

If you don't want the dashboard public, add a Cloudflare Access policy:

1. In Cloudflare Zero Trust → **Access** → **Applications** → **Add an application**
2. Choose **Self-hosted** → enter `fund.yourdomain.com`
3. Add a policy: **Allow** → **Emails** → `alexandercwesterberg@gmail.com`
4. Anyone visiting the URL will be prompted to verify their email before seeing the dashboard

---

## Troubleshooting

**Services won't start**
```bash
journalctl -u fundmgr-bot -n 50 --no-pager
journalctl -u fundmgr-web -n 50 --no-pager
```

**Deploy script fails**
```bash
tail -50 ~/ai-fund-manager/data/logs/deploy.log
```

**FinBERT model download is slow on first run**
The HuggingFace model (`ProsusAI/finbert`, ~440 MB) is downloaded on first use and cached in `~/.cache/huggingface/`. Subsequent runs are instant.

**yfinance rate-limited**
Price data is cached in SQLite for `lookback_days` (252 days). Re-running the same day re-uses the cache.

**`fund` command not found**
Make sure you're using the venv's binary: `~/.venv/bin/fund` or `source ~/.venv/bin/activate` first.
