import pandas as pd
import joblib
from pathlib import Path
from urllib.parse import urlparse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATASET_DIR = Path(r"C:\BLATS\cert_dataset")
MODEL_DIR = Path(r"C:\BLATS\backend\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HTTP_FILE = DATASET_DIR / "http.csv"
MODEL_FILE = MODEL_DIR / "network_model.pkl"

print("[+] Loading CERT http dataset...")
df = pd.read_csv(HTTP_FILE)

print("[+] Columns found:")
print(df.columns.tolist())

required_columns = ["id", "date", "user", "pc", "url"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["user"] = df["user"].astype(str)
df["pc"] = df["pc"].astype(str)
df["url"] = df["url"].astype(str)

def extract_domain(url):
    try:
        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc
        return url.split("/")[0]
    except Exception:
        return "unknown"

df["domain"] = df["url"].apply(extract_domain)

df["network_hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_after_hours"] = ((df["network_hour"] < 8) | (df["network_hour"] > 18)).astype(int)

df["date_only"] = df["date"].dt.date

df["daily_user_http_count"] = df.groupby(["user", "date_only"])["url"].transform("count")
df["daily_user_domain_count"] = df.groupby(["user", "date_only"])["domain"].transform("nunique")

df["user_domain_seen_before"] = df.groupby(["user", "domain"]).cumcount()
df["new_domain_for_user"] = (df["user_domain_seen_before"] == 0).astype(int)

df["url_length"] = df["url"].str.len()
df["domain_length"] = df["domain"].str.len()

features = [
    "network_hour",
    "day_of_week",
    "is_weekend",
    "is_after_hours",
    "daily_user_http_count",
    "daily_user_domain_count",
    "new_domain_for_user",
    "url_length",
    "domain_length",
    "pc",
    "domain"
]

X = df[features]

categorical_features = ["pc", "domain"]

numeric_features = [
    "network_hour",
    "day_of_week",
    "is_weekend",
    "is_after_hours",
    "daily_user_http_count",
    "daily_user_domain_count",
    "new_domain_for_user",
    "url_length",
    "domain_length"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

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

print("[+] Training network anomaly model...")
pipeline.fit(X)

joblib.dump(pipeline, MODEL_FILE)

print(f"[+] Network model saved to: {MODEL_FILE}")

all_preds = pipeline.predict(X)

normal_count = (all_preds == 1).sum()
anomaly_count = (all_preds == -1).sum()
total_count = len(all_preds)

print("[+] Training prediction summary:")
print(f"    Total rows   : {total_count}")
print(f"    Normal rows  : {normal_count}")
print(f"    Anomaly rows : {anomaly_count}")
print(f"    Anomaly rate : {(anomaly_count / total_count) * 100:.2f}%")

sample = X.sample(10, random_state=42)

scores = pipeline.decision_function(sample)
preds = pipeline.predict(sample)

print("[+] Random sample predictions:")

for i, pred in enumerate(preds):
    label = "NORMAL" if pred == 1 else "ANOMALY"
    print(f"    Row {i}: {label}, score={scores[i]:.6f}")

print("[+] Done.")