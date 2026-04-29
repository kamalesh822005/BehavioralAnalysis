import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "endpoint_model.pkl"

rng = np.random.default_rng(42)

# Normal behaviour: business hours Mon-Fri, routine sudo commands
N_NORMAL = 2000
normal = pd.DataFrame({
    "command_type": rng.choice(["sudo_list", "sudo_read", "sudo_service"], N_NORMAL),
    "hour": rng.integers(8, 18, N_NORMAL),
    "day_of_week": rng.integers(0, 5, N_NORMAL),
    "is_after_hours": np.zeros(N_NORMAL, dtype=int),
    "is_weekend": np.zeros(N_NORMAL, dtype=int),
    "daily_endpoint_count": rng.integers(1, 15, N_NORMAL),
    "is_firewall_change": np.zeros(N_NORMAL, dtype=int),
})

# Anomalous: off-hours, firewall changes, destructive commands, high frequency
N_ANOM = 100
anomalous = pd.DataFrame({
    "command_type": rng.choice(["sudo_delete", "sudo_modify", "sudo_firewall", "sudo_other"], N_ANOM),
    "hour": np.concatenate([rng.integers(0, 7, N_ANOM // 2), rng.integers(19, 24, N_ANOM // 2)]),
    "day_of_week": rng.integers(0, 7, N_ANOM),
    "is_after_hours": np.ones(N_ANOM, dtype=int),
    "is_weekend": rng.integers(0, 2, N_ANOM),
    "daily_endpoint_count": rng.integers(20, 80, N_ANOM),
    "is_firewall_change": rng.integers(0, 2, N_ANOM),
})

df = pd.concat([normal, anomalous], ignore_index=True)

categorical_features = ["command_type"]
numeric_features = [
    "hour", "day_of_week",
    "is_after_hours", "is_weekend",
    "daily_endpoint_count", "is_firewall_change",
]

X = df[categorical_features + numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", IsolationForest(n_estimators=200, contamination=0.05, random_state=42)),
])

print("[+] Training endpoint anomaly model...")
pipeline.fit(X)

joblib.dump(pipeline, MODEL_FILE)
print(f"[+] Endpoint model saved to: {MODEL_FILE}")

preds = pipeline.predict(X)
print(f"[+] Total: {len(preds)}  Normal: {(preds == 1).sum()}  Anomaly: {(preds == -1).sum()}")
