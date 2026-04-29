"""
Train all three BLATS anomaly models from the labeled CSV datasets
produced by build_dataset.py.

Usage
-----
    python train_from_dataset.py

If a CSV is missing or has fewer than 10 normal rows the script falls back
to built-in synthetic data so the backend can start immediately.

Reads  : C:\\BLATS\\dataset\\csv\\{login,endpoint,network}_dataset.csv
Writes : backend/models/{login,endpoint,network}_model.pkl
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATASET_DIR = Path(r"C:\BLATS\dataset\csv")
MODEL_DIR    = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_pipeline(cat_features, num_features, contamination=0.05):
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", IsolationForest(n_estimators=200,
                                  contamination=contamination,
                                  random_state=42)),
    ])


def _fit_and_save(name, X_train, X_all, y_all,
                  cat_features, num_features, contamination=0.05):
    print(f"\n[+] Training {name} model on {len(X_train):,} normal rows...")
    pipe = _build_pipeline(cat_features, num_features, contamination)
    pipe.fit(X_train)

    out = MODEL_DIR / f"{name}_model.pkl"
    joblib.dump(pipe, out)
    print(f"    Saved → {out}")

    preds = pipe.predict(X_all)
    if y_all is not None and int(y_all.sum()) > 0:
        tp = int(((preds == -1) & (y_all == 1)).sum())
        fp = int(((preds == -1) & (y_all == 0)).sum())
        fn = int(((preds ==  1) & (y_all == 1)).sum())
        print(f"    Labeled anomalies: {int(y_all.sum())}  "
              f"TP={tp}  FP={fp}  FN={fn}")
    else:
        print(f"    Flagged {int((preds == -1).sum())} of {len(preds)} rows "
              f"as anomalous (no labeled anomalies to validate against)")


# ──────────────────────────────────────────────────────────────────────────────
# LOGIN model
# Features: event_name (cat), login_hour, day_of_week,
#           is_weekend, is_after_hours, ssh_used
# Matches  : feature_mapper.py  (inference)
# ──────────────────────────────────────────────────────────────────────────────

CAT_LOGIN = ["event_name"]
NUM_LOGIN = ["login_hour", "day_of_week", "is_weekend", "is_after_hours", "ssh_used"]


def _coerce_login(df):
    df["event_name"]     = df["event_name"].fillna("unknown").astype(str)
    df["login_hour"]     = pd.to_numeric(df["login_hour"],     errors="coerce").fillna(0).astype(int)
    df["day_of_week"]    = pd.to_numeric(df["day_of_week"],    errors="coerce").fillna(0).astype(int)
    df["is_weekend"]     = pd.to_numeric(df["is_weekend"],     errors="coerce").fillna(0).astype(int)
    df["is_after_hours"] = pd.to_numeric(df["is_after_hours"], errors="coerce").fillna(0).astype(int)
    df["ssh_used"]       = pd.to_numeric(df["ssh_used"],       errors="coerce").fillna(0).astype(int)
    df["label"]          = pd.to_numeric(df.get("label", 0),   errors="coerce").fillna(0).astype(int)
    return df


def _synthetic_login():
    rng = np.random.default_rng(42)
    N = 2000
    normal = pd.DataFrame({
        "event_name":     rng.choice(["login_success", "gui_login_success", "gui_logout"], N),
        "login_hour":     rng.integers(8, 18, N),
        "day_of_week":    rng.integers(0, 5, N),
        "is_weekend":     np.zeros(N, dtype=int),
        "is_after_hours": np.zeros(N, dtype=int),
        "ssh_used":       rng.integers(0, 2, N),
        "label":          np.zeros(N, dtype=int),
    })
    anomalous = pd.DataFrame({
        "event_name":     rng.choice(["login_failed"] * 3 + ["login_success"], 200),
        "login_hour":     np.concatenate([rng.integers(0, 6, 100),
                                          rng.integers(21, 24, 100)]),
        "day_of_week":    rng.integers(0, 7, 200),
        "is_weekend":     rng.integers(0, 2, 200),
        "is_after_hours": np.ones(200, dtype=int),
        "ssh_used":       np.ones(200, dtype=int),
        "label":          np.ones(200, dtype=int),
    })
    return pd.concat([normal, anomalous], ignore_index=True)


def train_login():
    csv_path = DATASET_DIR / "login_dataset.csv"
    if csv_path.exists():
        df = _coerce_login(pd.read_csv(csv_path))
        print(f"[+] login_dataset.csv : {len(df):,} rows  "
              f"(anomalous: {int(df['label'].sum())})")
    else:
        print("[!] login_dataset.csv not found — using synthetic bootstrap")
        df = _coerce_login(_synthetic_login())

    X_all   = df[CAT_LOGIN + NUM_LOGIN].copy()
    y_all   = df["label"]
    X_train = X_all[y_all == 0]

    if len(X_train) < 10:
        print("    Fewer than 10 normal rows — padding with synthetic data")
        syn = _coerce_login(_synthetic_login())
        X_train = pd.concat([X_train,
                              syn[CAT_LOGIN + NUM_LOGIN][syn["label"] == 0]],
                             ignore_index=True)

    _fit_and_save("login", X_train, X_all, y_all, CAT_LOGIN, NUM_LOGIN)


# ──────────────────────────────────────────────────────────────────────────────
# NETWORK model
# Features: event_type, proto, app_proto (cat)
#           + ports, bytes, flow_age, alert_severity, hour, day_of_week,
#             is_alert, is_dns, is_http, is_tls, is_flow (num)
# Matches  : network_feature_mapper.py  (inference)
# ──────────────────────────────────────────────────────────────────────────────

CAT_NETWORK = ["event_type", "proto", "app_proto"]
NUM_NETWORK = [
    "src_port", "dest_port",
    "bytes_toserver", "bytes_toclient",
    "flow_age", "alert_severity",
    "hour", "day_of_week",
    "is_alert", "is_dns", "is_http", "is_tls", "is_flow",
]


def _coerce_network(df):
    for col in CAT_NETWORK:
        df[col] = df[col].fillna("unknown").astype(str)
    for col in ["src_port", "dest_port", "bytes_toserver", "bytes_toclient",
                "flow_age", "alert_severity", "hour", "day_of_week"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df["is_alert"] = (df["event_type"] == "alert").astype(int)
    df["is_dns"]   = (df["event_type"] == "dns").astype(int)
    df["is_http"]  = (df["event_type"] == "http").astype(int)
    df["is_tls"]   = (df["event_type"] == "tls").astype(int)
    df["is_flow"]  = (df["event_type"] == "flow").astype(int)
    df["label"]    = pd.to_numeric(df.get("label", 0), errors="coerce").fillna(0).astype(int)
    return df


def _synthetic_network():
    rng = np.random.default_rng(42)
    N = 3000
    normal = pd.DataFrame({
        "event_type":      rng.choice(["dns", "http", "flow"], N),
        "proto":           rng.choice(["TCP", "UDP"], N),
        "app_proto":       rng.choice(["dns", "http", "unknown"], N),
        "src_port":        rng.integers(1024, 65535, N),
        "dest_port":       rng.choice([53, 80, 443, 8080], N),
        "bytes_toserver":  rng.integers(100, 5_000, N).astype(float),
        "bytes_toclient":  rng.integers(200, 50_000, N).astype(float),
        "flow_age":        rng.integers(1, 300, N).astype(float),
        "alert_severity":  np.zeros(N),
        "hour":            rng.integers(7, 19, N),
        "day_of_week":     rng.integers(0, 5, N),
        "label":           np.zeros(N, dtype=int),
    })
    anomalous = pd.DataFrame({
        "event_type":      rng.choice(["alert", "flow", "dns"], 150),
        "proto":           rng.choice(["TCP", "UDP", "ICMP"], 150),
        "app_proto":       rng.choice(["unknown", "failed", "dns"], 150),
        "src_port":        rng.integers(1024, 65535, 150),
        "dest_port":       rng.choice([4444, 9090, 2222, 31337], 150),
        "bytes_toserver":  rng.integers(10_000_000, 100_000_000, 150).astype(float),
        "bytes_toclient":  rng.integers(1, 100, 150).astype(float),
        "flow_age":        rng.integers(3600, 86400, 150).astype(float),
        "alert_severity":  rng.integers(1, 4, 150).astype(float),
        "hour":            np.concatenate([rng.integers(0, 6, 75),
                                           rng.integers(21, 24, 75)]),
        "day_of_week":     rng.integers(0, 7, 150),
        "label":           np.ones(150, dtype=int),
    })
    return pd.concat([normal, anomalous], ignore_index=True)


def train_network():
    csv_path = DATASET_DIR / "network_dataset.csv"
    if csv_path.exists():
        df = _coerce_network(pd.read_csv(csv_path))
        print(f"[+] network_dataset.csv: {len(df):,} rows  "
              f"(anomalous: {int(df['label'].sum())})")
    else:
        print("[!] network_dataset.csv not found — using synthetic bootstrap")
        df = _coerce_network(_synthetic_network())

    X_all   = df[CAT_NETWORK + NUM_NETWORK].copy()
    y_all   = df["label"]
    X_train = X_all[y_all == 0]

    if len(X_train) < 10:
        print("    Fewer than 10 normal rows — padding with synthetic data")
        syn = _coerce_network(_synthetic_network())
        X_train = pd.concat([X_train,
                              syn[CAT_NETWORK + NUM_NETWORK][syn["label"] == 0]],
                             ignore_index=True)

    _fit_and_save("network", X_train, X_all, y_all,
                  CAT_NETWORK, NUM_NETWORK, contamination=0.02)


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT model
# Features: command_type (cat)
#           + login_hour, day_of_week, is_after_hours, is_weekend,
#             daily_endpoint_count, is_firewall_change (num)
# Matches  : endpoint_feature_mapper.py  (inference)
# ──────────────────────────────────────────────────────────────────────────────

_CMD_MAP = {
    "ls": "sudo_list",    "find": "sudo_list",
    "cat": "sudo_read",   "less": "sudo_read",  "more": "sudo_read",
    "tail": "sudo_read",  "head": "sudo_read",  "grep": "sudo_read",
    "service": "sudo_service", "systemctl": "sudo_service",
    "apt": "sudo_modify", "dpkg": "sudo_modify",
    "chmod": "sudo_modify", "chown": "sudo_modify",
    "mv": "sudo_modify",  "cp": "sudo_modify",
    "rm": "sudo_delete",
    "ufw": "sudo_firewall", "iptables": "sudo_firewall", "firewall": "sudo_firewall",
}

def _categorize_cmd(cmd: str) -> str:
    c = (cmd or "").lower()
    for kw, cat in _CMD_MAP.items():
        if kw in c:
            return cat
    return "sudo_other"


CAT_ENDPOINT = ["command_type"]
NUM_ENDPOINT = [
    "login_hour", "day_of_week", "is_after_hours",
    "is_weekend", "daily_endpoint_count", "is_firewall_change",
]


def _coerce_endpoint(df):
    if "command" in df.columns and "command_type" not in df.columns:
        fw = df.get("is_firewall_change", pd.Series(np.zeros(len(df)))).fillna(0).astype(int)
        df["command_type"] = [
            "sudo_firewall" if is_fw else _categorize_cmd(cmd)
            for cmd, is_fw in zip(df["command"].fillna(""), fw)
        ]
    df["command_type"]      = df.get("command_type", pd.Series(["sudo_other"] * len(df))).fillna("sudo_other").astype(str)
    df["is_firewall_change"] = pd.to_numeric(df.get("is_firewall_change", 0),
                                              errors="coerce").fillna(0).astype(int)
    for col in ["login_hour", "day_of_week", "is_after_hours", "is_weekend"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    # Compute daily_endpoint_count from (user_id, date) if columns exist
    if "timestamp" in df.columns and "user_id" in df.columns:
        df["_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        df["daily_endpoint_count"] = (
            df.groupby(["user_id", "_date"])["_date"]
              .transform("count")
              .fillna(1).astype(int)
        )
        df.drop(columns=["_date"], inplace=True)
    elif "daily_endpoint_count" not in df.columns:
        df["daily_endpoint_count"] = 1

    df["label"] = pd.to_numeric(df.get("label", 0), errors="coerce").fillna(0).astype(int)
    return df


def _synthetic_endpoint():
    rng = np.random.default_rng(42)
    N = 2000
    normal = pd.DataFrame({
        "command_type":        rng.choice(["sudo_list", "sudo_read", "sudo_service"], N),
        "login_hour":          rng.integers(8, 18, N),
        "day_of_week":         rng.integers(0, 5, N),
        "is_after_hours":      np.zeros(N, dtype=int),
        "is_weekend":          np.zeros(N, dtype=int),
        "daily_endpoint_count": rng.integers(1, 15, N),
        "is_firewall_change":  np.zeros(N, dtype=int),
        "label":               np.zeros(N, dtype=int),
    })
    anomalous = pd.DataFrame({
        "command_type":        rng.choice(["sudo_delete", "sudo_modify",
                                           "sudo_firewall", "sudo_other"], 100),
        "login_hour":          np.concatenate([rng.integers(0, 7, 50),
                                               rng.integers(19, 24, 50)]),
        "day_of_week":         rng.integers(0, 7, 100),
        "is_after_hours":      np.ones(100, dtype=int),
        "is_weekend":          rng.integers(0, 2, 100),
        "daily_endpoint_count": rng.integers(20, 80, 100),
        "is_firewall_change":  rng.integers(0, 2, 100),
        "label":               np.ones(100, dtype=int),
    })
    return pd.concat([normal, anomalous], ignore_index=True)


def train_endpoint():
    csv_path = DATASET_DIR / "endpoint_dataset.csv"
    if csv_path.exists():
        df = _coerce_endpoint(pd.read_csv(csv_path))
        print(f"[+] endpoint_dataset.csv: {len(df):,} rows  "
              f"(anomalous: {int(df['label'].sum())})")
    else:
        print("[!] endpoint_dataset.csv not found — using synthetic bootstrap")
        df = _coerce_endpoint(_synthetic_endpoint())

    X_all   = df[CAT_ENDPOINT + NUM_ENDPOINT].copy()
    y_all   = df["label"]
    X_train = X_all[y_all == 0]

    if len(X_train) < 10:
        print("    Fewer than 10 normal rows — padding with synthetic data")
        syn = _coerce_endpoint(_synthetic_endpoint())
        X_train = pd.concat([X_train,
                              syn[CAT_ENDPOINT + NUM_ENDPOINT][syn["label"] == 0]],
                             ignore_index=True)

    _fit_and_save("endpoint", X_train, X_all, y_all, CAT_ENDPOINT, NUM_ENDPOINT)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" BLATS Model Trainer")
    print(f" Dataset dir : {DATASET_DIR}")
    print(f" Model dir   : {MODEL_DIR}")
    print("=" * 60)

    train_login()
    train_network()
    train_endpoint()

    print("\n[+] All three models trained and saved.")
    print("    Next: uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
