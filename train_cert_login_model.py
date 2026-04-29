import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATASET_DIR = Path(r"C:\BLATS\cert_dataset")
MODEL_DIR = Path(r"C:\BLATS\backend\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOGON_FILE = DATASET_DIR / "logon.csv"
MODEL_FILE = MODEL_DIR / "login_model.pkl"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("[+] Loading CERT logon dataset...")

df = pd.read_csv(LOGON_FILE)

print("[+] Columns found:")
print(df.columns.tolist())

required_columns = ["id", "date", "user", "pc", "activity"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ---------------------------------------------------------
# Clean data
# ---------------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["user"] = df["user"].astype(str)
df["pc"] = df["pc"].astype(str)
df["activity"] = df["activity"].astype(str)

# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

df["login_hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# After-hours: before 8 AM or after 6 PM
df["is_after_hours"] = ((df["login_hour"] < 8) | (df["login_hour"] > 18)).astype(int)

# Daily count per user
df["date_only"] = df["date"].dt.date
df["daily_user_event_count"] = df.groupby(["user", "date_only"])["activity"].transform("count")

# First time user uses a PC
df["user_pc_seen_before"] = df.groupby(["user", "pc"]).cumcount()
df["new_pc_for_user"] = (df["user_pc_seen_before"] == 0).astype(int)

# ---------------------------------------------------------
# Features used by model
# ---------------------------------------------------------

features = [
    "login_hour",
    "day_of_week",
    "is_weekend",
    "is_after_hours",
    "daily_user_event_count",
    "new_pc_for_user",
    "activity",
    "pc"
]

X = df[features]

categorical_features = ["activity", "pc"]

numeric_features = [
    "login_hour",
    "day_of_week",
    "is_weekend",
    "is_after_hours",
    "daily_user_event_count",
    "new_pc_for_user"
]

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
# contamination=0.01 means model expects around 1% anomaly

model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("[+] Training login anomaly model...")
pipeline.fit(X)

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

joblib.dump(pipeline, MODEL_FILE)

print(f"[+] Login model saved to: {MODEL_FILE}")

# ---------------------------------------------------------
# Overall prediction summary
# ---------------------------------------------------------

all_preds = pipeline.predict(X)

normal_count = (all_preds == 1).sum()
anomaly_count = (all_preds == -1).sum()
total_count = len(all_preds)

print("[+] Training prediction summary:")
print(f"    Total rows   : {total_count}")
print(f"    Normal rows  : {normal_count}")
print(f"    Anomaly rows : {anomaly_count}")
print(f"    Anomaly rate : {(anomaly_count / total_count) * 100:.2f}%")

# ---------------------------------------------------------
# Random sample test
# ---------------------------------------------------------

sample = X.sample(10, random_state=42)

scores = pipeline.decision_function(sample)
preds = pipeline.predict(sample)

print("[+] Random sample predictions:")

for i, pred in enumerate(preds):
    label = "NORMAL" if pred == 1 else "ANOMALY"
    print(f"    Row {i}: {label}, score={scores[i]:.6f}")

print("[+] Done.")