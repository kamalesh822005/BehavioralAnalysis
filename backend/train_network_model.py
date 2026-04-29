import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_FILE = Path(__file__).parent / "network_features.json"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "network_model.pkl"

rows = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)

df["event_type"] = df["event_type"].fillna("unknown")
df["proto"] = df["proto"].fillna("unknown")
df["app_proto"] = df["app_proto"].fillna("unknown")
df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0)
df["dest_port"] = pd.to_numeric(df["dest_port"], errors="coerce").fillna(0)
df["bytes_toserver"] = pd.to_numeric(df["bytes_toserver"], errors="coerce").fillna(0)
df["bytes_toclient"] = pd.to_numeric(df["bytes_toclient"], errors="coerce").fillna(0)
df["flow_age"] = pd.to_numeric(df["flow_age"], errors="coerce").fillna(0)
df["alert_severity"] = pd.to_numeric(df["alert_severity"], errors="coerce").fillna(0)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)

df["is_alert"] = (df["event_type"] == "alert").astype(int)
df["is_dns"] = (df["event_type"] == "dns").astype(int)
df["is_http"] = (df["event_type"] == "http").astype(int)
df["is_tls"] = (df["event_type"] == "tls").astype(int)
df["is_flow"] = (df["event_type"] == "flow").astype(int)

categorical_features = ["event_type", "proto", "app_proto"]
numeric_features = [
    "src_port", "dest_port",
    "bytes_toserver", "bytes_toclient",
    "flow_age", "alert_severity",
    "hour", "day_of_week",
    "is_alert", "is_dns", "is_http", "is_tls", "is_flow",
]

X = df[categorical_features + numeric_features].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", IsolationForest(n_estimators=200, contamination=0.02, random_state=42)),
])

print("[+] Training network anomaly model (IsolationForest)...")
pipeline.fit(X)

joblib.dump(pipeline, MODEL_FILE)
print(f"[+] Network model saved to: {MODEL_FILE}")

preds = pipeline.predict(X)
print(f"[+] Total: {len(preds)}  Normal: {(preds == 1).sum()}  Anomaly: {(preds == -1).sum()}")
