from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict
from pathlib import Path
import joblib

from feature_mapper import map_event_to_features

app = FastAPI(title="BLATS Backend")

MODEL_PATH = Path(r"C:\BLATS\backend\models\login_model.pkl")

login_model = joblib.load(MODEL_PATH)

users = {}
events = []


class LogstashEvent(BaseModel):
    event_name: Optional[str] = None
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    source_type: Optional[str] = None
    blats_category: Optional[str] = None
    severity: Optional[str] = None
    trust_impact: Optional[int] = 0
    frontend_message: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    class Config:
        extra = "allow"


def rule_risk(event: Dict[str, Any]) -> int:
    event_name = event.get("event_name")

    if event_name == "login_failed":
        return 45
    if event_name == "login_success":
        return 5
    if event_name == "gui_login_success":
        return 5
    if event_name == "gui_logout":
        return 0
    if event_name == "sudo_command":
        return 55
    if event_name == "firewall_change":
        return 75
    if event_name == "alert":
        return 85
    if event_name in ["dns", "http", "flow"]:
        return 10

    return 5


def ml_login_risk(event: Dict[str, Any]) -> int:
    event_name = event.get("event_name")

    if event_name not in ["login_success", "gui_login_success", "gui_logout"]:
        return 0

    features = map_event_to_features(event)

    prediction = login_model.predict(features)[0]
    score = login_model.decision_function(features)[0]

    if prediction == -1:
        return 60

    return 5


def update_trust(user_id: str, final_risk: int, event: Dict[str, Any]) -> int:
    if user_id not in users:
        users[user_id] = {
            "trust_score": 100,
            "status": "trusted",
            "event_count": 0,
            "last_event": None,
            "last_risk": 0
        }

    user = users[user_id]

    trust_impact = event.get("trust_impact", 0) or 0

    if final_risk >= 80:
        trust_change = -25
    elif final_risk >= 60:
        trust_change = -15
    elif final_risk >= 40:
        trust_change = -8
    elif final_risk <= 10:
        trust_change = trust_impact
    else:
        trust_change = -2

    new_score = user["trust_score"] + trust_change
    new_score = max(0, min(100, new_score))

    user["trust_score"] = new_score
    user["event_count"] += 1
    user["last_event"] = event.get("event_name")
    user["last_risk"] = final_risk

    if new_score >= 75:
        user["status"] = "trusted"
    elif new_score >= 40:
        user["status"] = "watch"
    else:
        user["status"] = "risky"

    return new_score


@app.get("/")
def home():
    return {
        "status": "BLATS backend running",
        "model_loaded": MODEL_PATH.exists()
    }


@app.post("/events")
def receive_event(event: LogstashEvent):
    event_dict = event.dict()

    user_id = event.user_id or "unknown"

    r_risk = rule_risk(event_dict)
    m_risk = ml_login_risk(event_dict)

    final_risk = int((0.5 * r_risk) + (0.5 * m_risk))

    trust_score = update_trust(user_id, final_risk, event_dict)

    output = {
        "user_id": user_id,
        "event_name": event.event_name,
        "category": event.blats_category,
        "severity": event.severity,
        "rule_risk": r_risk,
        "ml_risk": m_risk,
        "final_risk": final_risk,
        "trust_score": trust_score,
        "frontend_message": event.frontend_message,
        "source_ip": event.source_ip,
        "raw_message": event.message,
        "timestamp": event.timestamp,
    }

    events.append(output)

    return {
        "status": "received",
        "result": output
    }


@app.get("/users")
def get_users():
    return users


@app.get("/events")
def get_events():
    return events[-100:]


@app.get("/dashboard")
def dashboard():
    return {
        "total_users": len(users),
        "total_events": len(events),
        "users": users,
        "recent_events": events[-20:]
    }