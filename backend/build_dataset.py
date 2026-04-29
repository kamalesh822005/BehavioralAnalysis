"""
Reads raw JSONL files written by Logstash and produces labeled CSVs
for training the login, network, and endpoint models.

Normal events  → label = 0
Anomalous events → label = 1  (set ATTACK_WINDOW below after running simulate_attacks.sh)

Usage:
    python build_dataset.py
    python build_dataset.py --attack-start "2026-04-30 02:00" --attack-end "2026-04-30 02:30"
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

DATASET_DIR = Path(r"C:\BLATS\dataset")
OUTPUT_DIR = DATASET_DIR / "csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[!] Not found: {path}")
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[+] Loaded {len(rows):,} rows from {path.name}")
    return rows


def label_rows(rows: list[dict], attack_start, attack_end) -> list[dict]:
    """Add label=1 for events inside the attack window, label=0 otherwise."""
    for row in rows:
        ts = pd.to_datetime(row.get("@timestamp") or row.get("timestamp"), errors="coerce")
        if attack_start and attack_end and pd.notna(ts):
            row["label"] = 1 if attack_start <= ts.tz_localize(None) <= attack_end else 0
        else:
            row["label"] = 0
    return rows


def build_login_csv(rows: list[dict]) -> pd.DataFrame:
    login_events = {"login_failed", "login_success", "gui_login_success", "gui_logout"}
    records = []
    for r in rows:
        if r.get("event_name") not in login_events:
            continue
        ts = pd.to_datetime(r.get("@timestamp") or r.get("timestamp"), errors="coerce")
        records.append({
            "timestamp": ts,
            "user_id": r.get("user_id", "unknown"),
            "event_name": r.get("event_name"),
            "source_ip": r.get("source_ip", ""),
            "login_hour": ts.hour if pd.notna(ts) else -1,
            "day_of_week": ts.dayofweek if pd.notna(ts) else -1,
            "is_weekend": int(ts.dayofweek in (5, 6)) if pd.notna(ts) else 0,
            "is_after_hours": int(ts.hour < 8 or ts.hour > 18) if pd.notna(ts) else 0,
            "ssh_used": int(bool(r.get("ssh_used"))),
            "label": r.get("label", 0),
        })
    return pd.DataFrame(records)


def build_endpoint_csv(rows: list[dict]) -> pd.DataFrame:
    endpoint_events = {"sudo_command", "firewall_change"}
    records = []
    for r in rows:
        if r.get("event_name") not in endpoint_events:
            continue
        ts = pd.to_datetime(r.get("@timestamp") or r.get("timestamp"), errors="coerce")
        command = r.get("command") or r.get("message", "")
        records.append({
            "timestamp": ts,
            "user_id": r.get("user_id", "unknown"),
            "event_name": r.get("event_name"),
            "command": command,
            "target_user": r.get("target_user", ""),
            "login_hour": ts.hour if pd.notna(ts) else -1,
            "day_of_week": ts.dayofweek if pd.notna(ts) else -1,
            "is_after_hours": int(ts.hour < 8 or ts.hour > 18) if pd.notna(ts) else 0,
            "is_weekend": int(ts.dayofweek in (5, 6)) if pd.notna(ts) else 0,
            "is_firewall_change": int(r.get("event_name") == "firewall_change"),
            "label": r.get("label", 0),
        })
    return pd.DataFrame(records)


def build_network_csv(rows: list[dict]) -> pd.DataFrame:
    network_events = {"dns", "http", "flow", "alert", "tls", "fileinfo"}
    records = []
    for r in rows:
        if r.get("event_name") not in network_events:
            continue
        ts = pd.to_datetime(r.get("@timestamp") or r.get("timestamp"), errors="coerce")
        records.append({
            "timestamp": ts,
            "event_type": r.get("event_name", "flow"),
            "src_ip": r.get("src_ip", ""),
            "dest_ip": r.get("dest_ip", ""),
            "src_port": r.get("src_port", 0),
            "dest_port": r.get("dest_port", 0),
            "proto": r.get("proto", "unknown"),
            "app_proto": r.get("app_proto", "unknown") or "unknown",
            "bytes_toserver": r.get("bytes_toserver", 0) or 0,
            "bytes_toclient": r.get("bytes_toclient", 0) or 0,
            "flow_age": r.get("flow_age", 0) or 0,
            "alert_severity": r.get("alert", {}).get("severity", 0) if isinstance(r.get("alert"), dict) else 0,
            "hour": ts.hour if pd.notna(ts) else -1,
            "day_of_week": ts.dayofweek if pd.notna(ts) else -1,
            "label": r.get("label", 0),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-start", help="Start of attack window e.g. '2026-04-30 02:00'")
    parser.add_argument("--attack-end", help="End of attack window e.g. '2026-04-30 02:30'")
    args = parser.parse_args()

    attack_start = pd.to_datetime(args.attack_start) if args.attack_start else None
    attack_end = pd.to_datetime(args.attack_end) if args.attack_end else None

    if attack_start and attack_end:
        print(f"[+] Attack window: {attack_start} → {attack_end}")
    else:
        print("[!] No attack window set — all events labeled normal (label=0)")
        print("    Re-run with --attack-start and --attack-end after simulating attacks")

    login_rows  = label_rows(load_jsonl(DATASET_DIR / "raw_login.jsonl"), attack_start, attack_end)
    endpoint_rows = label_rows(load_jsonl(DATASET_DIR / "raw_endpoint.jsonl"), attack_start, attack_end)
    network_rows = label_rows(load_jsonl(DATASET_DIR / "raw_network.jsonl"), attack_start, attack_end)

    login_df    = build_login_csv(login_rows + endpoint_rows)
    endpoint_df = build_endpoint_csv(endpoint_rows)
    network_df  = build_network_csv(network_rows)

    login_out    = OUTPUT_DIR / "login_dataset.csv"
    endpoint_out = OUTPUT_DIR / "endpoint_dataset.csv"
    network_out  = OUTPUT_DIR / "network_dataset.csv"

    login_df.to_csv(login_out, index=False)
    endpoint_df.to_csv(endpoint_out, index=False)
    network_df.to_csv(network_out, index=False)

    print(f"\n[+] login_dataset.csv    → {len(login_df):,} rows  (anomalous: {login_df['label'].sum()})")
    print(f"[+] endpoint_dataset.csv → {len(endpoint_df):,} rows  (anomalous: {endpoint_df['label'].sum()})")
    print(f"[+] network_dataset.csv  → {len(network_df):,} rows  (anomalous: {network_df['label'].sum()})")
    print(f"\n[+] CSVs saved to: {OUTPUT_DIR}")
