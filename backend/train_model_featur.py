import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "network_features.json"
MODEL_FILE = "models/network_model.pkl"

rows = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)

df["app_proto"] = df["app_proto"].fillna("unknown")
df["alert_signature"] = df["alert_signature"].fillna("")
df["src_ip"] = df["src_ip"].fillna("unknown")
df["dest_ip"] = df["dest_ip"].fillna("unknown")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour.fillna(0)
df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0)

df["is_alert"] = (df["event_type"] == "alert").astype(int)
df["is_dns"] = (df["event_type"] == "dns").astype(int)
df["is_http"] = (df["event_type"] == "http").astype(int)
df["is_tls"] = (df["event_type"] == "tls").astype(int)
df["is_flow"] = (df["event_type"] == "flow").astype(int)

cat_cols = ["event_type", "proto", "app_proto"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

features = [
    "event_type",
    "src_port",
    "dest_port",
    "proto",
    "app_proto",
    "bytes_toserver",
    "bytes_toclient",
    "flow_age",
    "alert_severity",
    "hour",
    "day_of_week",
    "is_alert",
    "is_dns",
    "is_http",
    "is_tls",
    "is_flow",
]

X = df[features].fillna(0)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(
    {
        "model": model,
        "features": features,
        "encoders": encoders
    },
    MODEL_FILE
)

print(f"[+] Saved network model to {MODEL_FILE}")