"""
Populates the CSV dataset files with realistic synthetic normal behavior
so the models have enough data to train on.
Appends to existing real rows rather than replacing them.
"""
import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(r"C:\BLATS\dataset\csv")
OUT.mkdir(parents=True, exist_ok=True)

USERS = ["ubuntu", "kamal", "admin", "devuser"]
IPS   = ["192.168.1.46", "192.168.1.50", "192.168.1.55", "127.0.0.1"]

# ── LOGIN ─────────────────────────────────────────────────────────────────────
N = 1000
ev_pool = (["login_success"] * 4 + ["gui_login_success"] * 3
           + ["gui_logout"] * 2 + ["login_failed"] * 1)
events  = rng.choice(ev_pool, N)
ssh     = np.array([1 if e in ("login_success", "login_failed") else 0
                    for e in events])

login_syn = pd.DataFrame({
    "timestamp":     pd.date_range("2026-01-01", periods=N, freq="1h"),
    "user_id":       rng.choice(USERS, N),
    "event_name":    events,
    "source_ip":     rng.choice(IPS, N),
    "login_hour":    rng.integers(8, 18, N),
    "day_of_week":   rng.integers(0, 5, N),
    "is_weekend":    np.zeros(N, dtype=int),
    "is_after_hours": np.zeros(N, dtype=int),
    "ssh_used":      ssh,
    "label":         np.zeros(N, dtype=int),
})

login_path = OUT / "login_dataset.csv"
if login_path.exists():
    real = pd.read_csv(login_path)
    combined = pd.concat([real, login_syn], ignore_index=True)
else:
    combined = login_syn
combined.to_csv(login_path, index=False)
print(f"login_dataset.csv    : {len(combined)} rows")

# ── ENDPOINT ──────────────────────────────────────────────────────────────────
N = 800
COMMANDS = [
    "sudo ls /home", "sudo cat /etc/hostname",
    "sudo systemctl status ssh", "sudo apt update",
    "sudo find /var/log -name *.log", "sudo tail /var/log/syslog",
    "sudo grep error /var/log/auth.log", "sudo service nginx status",
    "sudo dpkg -l", "sudo chown ubuntu /tmp/test",
    "sudo head /etc/passwd", "sudo systemctl list-units",
]

ep_syn = pd.DataFrame({
    "timestamp":         pd.date_range("2026-01-01", periods=N, freq="30min"),
    "user_id":           rng.choice(USERS, N),
    "event_name":        ["sudo_command"] * N,
    "command":           rng.choice(COMMANDS, N),
    "target_user":       ["root"] * N,
    "login_hour":        rng.integers(8, 18, N),
    "day_of_week":       rng.integers(0, 5, N),
    "is_after_hours":    np.zeros(N, dtype=int),
    "is_weekend":        np.zeros(N, dtype=int),
    "is_firewall_change": np.zeros(N, dtype=int),
    "label":             np.zeros(N, dtype=int),
})

ep_path = OUT / "endpoint_dataset.csv"
if ep_path.exists():
    real_ep = pd.read_csv(ep_path)
    combined_ep = pd.concat([real_ep, ep_syn], ignore_index=True)
else:
    combined_ep = ep_syn
combined_ep.to_csv(ep_path, index=False)
print(f"endpoint_dataset.csv : {len(combined_ep)} rows")

net_path = OUT / "network_dataset.csv"
net_rows = len(pd.read_csv(net_path)) if net_path.exists() else 0
print(f"network_dataset.csv  : {net_rows} rows (unchanged)")

print("\nDone. Now run: python train_from_dataset.py")
