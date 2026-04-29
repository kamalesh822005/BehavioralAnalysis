from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict
from pathlib import Path
import joblib

from feature_mapper import map_event_to_features
from network_feature_mapper import map_network_event_to_features
from endpoint_feature_mapper import map_endpoint_event_to_features

app = FastAPI(title="BLATS Backend")

_MODEL_DIR = Path(__file__).parent / "models"

login_model = joblib.load(_MODEL_DIR / "login_model.pkl")

_network_model_path = _MODEL_DIR / "network_model.pkl"
network_model = joblib.load(_network_model_path) if _network_model_path.exists() else None

_endpoint_model_path = _MODEL_DIR / "endpoint_model.pkl"
endpoint_model = joblib.load(_endpoint_model_path) if _endpoint_model_path.exists() else None

users: Dict[str, Any] = {}
events: list = []

_LOGIN_EVENTS = {"login_success", "gui_login_success", "gui_logout", "login_failed"}
_NETWORK_EVENTS = {"dns", "http", "flow", "alert", "tls"}
_ENDPOINT_EVENTS = {"sudo_command", "firewall_change"}


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


# ---------------------------------------------------------------------------
# Rule-based risk
# ---------------------------------------------------------------------------

_RULE_RISK_TABLE = {
    "login_failed": 45,
    "login_success": 5,
    "gui_login_success": 5,
    "gui_logout": 0,
    "sudo_command": 55,
    "firewall_change": 75,
    "alert": 85,
    "dns": 10,
    "http": 10,
    "flow": 10,
}


def rule_risk(event: Dict[str, Any]) -> int:
    return _RULE_RISK_TABLE.get(event.get("event_name"), 5)


# ---------------------------------------------------------------------------
# ML risk per category
# ---------------------------------------------------------------------------

def ml_login_risk(event: Dict[str, Any]) -> int:
    if event.get("event_name") not in _LOGIN_EVENTS:
        return 0
    try:
        features = map_event_to_features(event)
        pred = login_model.predict(features)[0]
        return 60 if pred == -1 else 5
    except Exception:
        return 0


def ml_network_risk(event: Dict[str, Any]) -> int:
    if event.get("event_name") not in _NETWORK_EVENTS or network_model is None:
        return 0
    try:
        features = map_network_event_to_features(event)
        pred = network_model.predict(features)[0]
        return 65 if pred == -1 else 5
    except Exception:
        return 0


def ml_endpoint_risk(event: Dict[str, Any]) -> int:
    if event.get("event_name") not in _ENDPOINT_EVENTS or endpoint_model is None:
        return 0
    try:
        features = map_endpoint_event_to_features(event)
        pred = endpoint_model.predict(features)[0]
        return 70 if pred == -1 else 10
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Trust score engine
# ---------------------------------------------------------------------------

def update_trust(user_id: str, final_risk: int, event: Dict[str, Any]) -> int:
    if user_id not in users:
        users[user_id] = {
            "trust_score": 100,
            "status": "trusted",
            "event_count": 0,
            "last_event": None,
            "last_risk": 0,
        }

    user = users[user_id]
    trust_impact = event.get("trust_impact") or 0

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

    new_score = max(0, min(100, user["trust_score"] + trust_change))
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "BLATS backend running",
        "models": {
            "login": True,
            "network": network_model is not None,
            "endpoint": endpoint_model is not None,
        },
    }


@app.post("/events")
def receive_event(event: LogstashEvent):
    event_dict = event.dict()
    user_id = event.user_id or "unknown"

    r_risk = rule_risk(event_dict)
    m_login = ml_login_risk(event_dict)
    m_network = ml_network_risk(event_dict)
    m_endpoint = ml_endpoint_risk(event_dict)

    # Only non-zero ML score contributes; rule weight 50%, ML weight 50%
    ml_risk = m_login or m_network or m_endpoint
    final_risk = int(0.5 * r_risk + 0.5 * ml_risk)

    trust_score = update_trust(user_id, final_risk, event_dict)

    output = {
        "user_id": user_id,
        "event_name": event.event_name,
        "category": event.blats_category,
        "severity": event.severity,
        "rule_risk": r_risk,
        "ml_login_risk": m_login,
        "ml_network_risk": m_network,
        "ml_endpoint_risk": m_endpoint,
        "final_risk": final_risk,
        "trust_score": trust_score,
        "frontend_message": event.frontend_message,
        "source_ip": event.source_ip,
        "raw_message": event.message,
        "timestamp": event.timestamp,
    }

    events.append(output)
    return {"status": "received", "result": output}


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
        "recent_events": events[-20:],
    }
