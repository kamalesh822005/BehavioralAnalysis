#!/bin/bash
# Run this on the Ubuntu VM to generate anomalous events.
# Copy the START/END timestamps it prints into build_dataset.py --attack-start / --attack-end

echo "============================================"
echo " BLATS Attack Simulator"
echo " Run on: Ubuntu VM"
echo "============================================"
echo ""
echo "[*] Attack window START: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── 1. Brute-force SSH logins ────────────────────────────────────────────────
echo "[*] Simulating SSH brute-force (failed logins)..."
for i in {1..15}; do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
      baduser@127.0.0.1 exit 2>/dev/null || true
  sleep 0.5
done

# ── 2. Suspicious sudo commands ──────────────────────────────────────────────
echo "[*] Simulating suspicious sudo commands..."
sudo ls /root 2>/dev/null || true
sudo cat /etc/shadow 2>/dev/null || true
sudo find /home -name "*.txt" 2>/dev/null || true
sudo cat /etc/passwd 2>/dev/null || true

# ── 3. Firewall changes ──────────────────────────────────────────────────────
echo "[*] Simulating firewall changes..."
sudo ufw deny 9090 2>/dev/null || true
sudo ufw deny 4444 2>/dev/null || true
sudo ufw allow 2222 2>/dev/null || true
# Undo immediately so the VM isn't affected
sudo ufw delete deny 9090 2>/dev/null || true
sudo ufw delete deny 4444 2>/dev/null || true
sudo ufw delete allow 2222 2>/dev/null || true

# ── 4. Large file download (simulates data exfiltration via network) ──────────
echo "[*] Simulating large download (network anomaly)..."
wget -q --timeout=15 -O /dev/null http://speedtest.tele2.net/10MB.zip 2>/dev/null || \
curl -s --max-time 15 -o /dev/null http://speedtest.tele2.net/10MB.zip 2>/dev/null || \
echo "    (download skipped — no internet access)"

# ── 5. Unusual outbound connections (triggers Suricata flow/dns events) ───────
echo "[*] Simulating unusual outbound connections..."
curl -s --max-time 5 -o /dev/null http://checkip.amazonaws.com 2>/dev/null || true
curl -s --max-time 5 -o /dev/null http://ifconfig.me 2>/dev/null || true
nslookup pastebin.com 2>/dev/null || true
nslookup raw.githubusercontent.com 2>/dev/null || true

# ── 6. Port scan (triggers Suricata alert) ────────────────────────────────────
echo "[*] Simulating port scan (if nmap installed)..."
if command -v nmap &>/dev/null; then
  sudo nmap -sS -p 22,80,443,3306,5432 127.0.0.1 2>/dev/null || true
else
  echo "    (nmap not installed — skipping)"
fi

echo ""
echo "[*] Attack window END: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " Done. Now run on Windows:"
echo ""
echo "   cd C:\\BLATS\\backend"
echo "   python build_dataset.py \\"
echo "     --attack-start \"<START above>\" \\"
echo "     --attack-end   \"<END above>\""
echo "============================================"
