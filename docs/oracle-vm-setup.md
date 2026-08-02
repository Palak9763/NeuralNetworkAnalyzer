# Oracle Cloud Always Free VM - Deployment Guide

This guide covers the one-time host-level setup for running the
NeuralNetworkAnalyzer backend on an Oracle Cloud Always Free VM
(2 OCPU, 12 GB RAM, Ubuntu 22.04).

---

## Prerequisites

- Oracle Cloud account with an Always Free VM provisioned
- SSH access to the VM (`ssh ubuntu@<vm-public-ip>`)
- A domain name with a DNS A record pointed at the VM public IP
  (required for automatic HTTPS via Caddy / Let us Encrypt)
- Docker and Docker Compose installed on the VM

---

## Step 1: Install Docker

```bash
# Update packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add your user to the docker group so you don't need sudo every time
sudo usermod -aG docker $USER

# Apply group change (or log out and back in)
newgrp docker

# Install Docker Compose plugin
sudo apt-get install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

---

## Step 2: Create a 4 GB Swap File (memory safety net)

With 12 GB RAM and an 8g mem_limit on the backend container, swap acts as
a buffer for brief transient spikes beyond the hard limit before Docker's
OOM killer fires.

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
# Should show ~4G in the Swap row
```

---

## Step 3: Configure the Firewall (BOTH layers required)

Oracle Cloud VMs have TWO independent firewall layers. Both must be open
or external connections will never reach the VM, even if the app is running.

### 3a. OS-level firewall (iptables)

```bash
# Allow HTTP and HTTPS
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT

# Persist across reboots
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save
```

### 3b. Oracle Cloud VCN Security List (OCI Console)

1. Log in to https://cloud.oracle.com
2. Navigate to: Networking -> Virtual Cloud Networks -> your VCN ->
   Security Lists -> Default Security List
3. Click "Add Ingress Rules" and add:
   - Source CIDR: `0.0.0.0/0`, Protocol: TCP, Destination Port Range: `80`
   - Source CIDR: `0.0.0.0/0`, Protocol: TCP, Destination Port Range: `443`

**Verification test**: From your LOCAL machine (not from inside the VM):
```bash
curl http://<vm-public-ip>
# If the Security List rule is missing, this will time out.
# If only iptables is missing, you get "connection refused" immediately.
# Both passing means both layers are correctly open.
```

---

## Step 4: Clone the Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/NeuralNetworkAnalyzer.git
cd NeuralNetworkAnalyzer
```

---

## Step 5: Create the Production Environment File

**Never commit this file.** It is excluded by .gitignore.

```bash
nano backend/.env.production
```

Paste and fill in real values:

```env
NNA_DATABASE_URL=postgresql://postgres.YOUR_PROJECT_ID:YOUR_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
NNA_JWT_SECRET_KEY=<run: openssl rand -hex 32>
NNA_CORS_ORIGINS=https://your-vercel-frontend.vercel.app,https://your.domain.com
NNA_GOOGLE_CLIENT_ID=440864790867-jner1beitg78g3hi9shiaarrkbpa0dd3.apps.googleusercontent.com
NNA_GOOGLE_CLIENT_SECRET=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
NNA_GITHUB_CLIENT_ID=
NNA_GITHUB_CLIENT_SECRET=
```

---

## Step 6: Configure Caddy

Edit `Caddyfile` in the repo root and replace `your.domain.com` with your
actual domain:

```bash
sed -i 's/your.domain.com/api.yourdomain.com/' Caddyfile
```

Verify the DNS A record is live before starting Caddy:

```bash
dig +short api.yourdomain.com
# Should return the VM's public IP
```

---

## Step 7: Build and Start the Stack

```bash
# Build the Docker image and start all services
docker compose up -d --build

# Follow logs to confirm startup
docker compose logs -f backend
# Wait until you see: "Application startup complete"

# Check container resource usage
docker stats --no-stream
```

**Expected baseline memory** (before any request): `< 400 MB RSS`.
This confirms lazy ML imports are working correctly -- the ML libraries are
NOT loaded at startup, only when a parse request arrives for that framework.

---

## Step 8: Verify Health

```bash
# From inside the VM
curl http://localhost:8000/api/v1/health
# Expected: {"status":"ok"}

# From your local machine (tests BOTH firewall layers)
curl https://your.domain.com/api/v1/health
# Expected: {"status":"ok"}
```

---

## Step 9: Enable Auto-Start on VM Reboot

```bash
sudo nano /etc/systemd/system/nna.service
```

Paste:

```ini
[Unit]
Description=NeuralNetworkAnalyzer Docker Compose Stack
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/NeuralNetworkAnalyzer
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

Enable and test:

```bash
sudo systemctl daemon-reload
sudo systemctl enable nna.service
sudo systemctl start nna.service
sudo systemctl status nna.service
```

**Test the reboot path explicitly:**

```bash
sudo reboot
# Wait 2-3 minutes, then from your local machine:
curl https://your.domain.com/api/v1/health
# Must return {"status":"ok"} with no manual intervention.
```

---

## Testing Checklist

### 1. Baseline memory (lazy-import verification)

```bash
docker stats --no-stream
# EXPECTED: backend container < 400 MB before any request
# If it shows 2+ GB, the lazy import audit failed somewhere
```

### 2. Health check

```bash
curl https://your.domain.com/api/v1/health
# Expected: {"status": "ok"}
```

### 3. Peak memory during heavy parse

Upload a HuggingFace model or a large TensorFlow model. Watch live:

```bash
watch -n 2 "docker stats --no-stream"
# Record the peak MEM USAGE for backend. Should stay under 8 GiB (mem_limit).
```

### 4. OOM-kill-and-restart test

```bash
# Temporarily set mem_limit to something clearly insufficient
sed -i 's/mem_limit: 8g/mem_limit: 200m/' docker-compose.yml
docker compose up -d
# Send a parse request -> container should be killed and restart automatically
docker compose logs backend  # look for "Restarting" events
# Restore the real limit
sed -i 's/mem_limit: 200m/mem_limit: 8g/' docker-compose.yml
docker compose up -d
```

### 5. Gunicorn worker recycling (memory floor check)

Run 60+ mixed PyTorch + TF parse requests. Monitor memory:

```bash
watch -n 5 "docker stats --no-stream"
# Around request 50, memory should drop noticeably (worker recycled)
# Over 60 requests the overall trend should be flat, not monotonically rising
```

### 6. Upload cleanup verification

```bash
# Create a fake job dir older than 24 hours
mkdir -p backend/storage/uploads/test_old_job
touch -t $(date -d "25 hours ago" +"%Y%m%d%H%M") backend/storage/uploads/test_old_job

# Check docker logs after the next sweep (up to 1 hour)
docker compose logs backend | grep "upload_cleanup"
# Should see: "Removed stale upload dir: test_old_job"
```

### 7. VM reboot auto-start test

See Step 9 above.

### 8. Dual-firewall external connectivity test

```bash
# Run from YOUR LOCAL MACHINE, not from inside the VM
curl -v https://your.domain.com/api/v1/health
# Check: no timeout, no SSL errors, {"status":"ok"} in body
```

### 9. Full end-to-end from Vercel frontend

1. Open the Vercel-deployed frontend
2. Upload a .py model file
3. Confirm the diagram renders
4. Open browser DevTools -> Network tab
5. Verify: no CORS errors, requests go to `https://your.domain.com` (not localhost)
6. Repeat with a TensorFlow model specifically (highest memory footprint)

---

## Maintenance

### View logs

```bash
docker compose logs -f backend          # stream live
docker compose logs --tail=100 backend  # last 100 lines
```

### Update and redeploy

```bash
git pull
docker compose up -d --build
```

### Check container stats

```bash
docker stats --no-stream
```

### Manually trigger upload cleanup (for testing)

```bash
docker compose exec backend python -c "
import time, shutil
from pathlib import Path
from app.core.config import settings

cutoff = time.time() - 24 * 3600
for d in settings.upload_dir.iterdir():
    if d.is_dir() and d.stat().st_mtime < cutoff:
        print(f'Would delete: {d}')
        # shutil.rmtree(d)  # uncomment to actually delete
"
```
