import pandas as pd
from datetime import datetime

_user_daily_counts: dict = {}  # (user, date) -> int

_COMMAND_CATEGORIES = {
    "ls": "sudo_list",
    "find": "sudo_list",
    "cat": "sudo_read",
    "less": "sudo_read",
    "more": "sudo_read",
    "tail": "sudo_read",
    "head": "sudo_read",
    "grep": "sudo_read",
    "service": "sudo_service",
    "systemctl": "sudo_service",
    "apt": "sudo_modify",
    "dpkg": "sudo_modify",
    "chmod": "sudo_modify",
    "chown": "sudo_modify",
    "mv": "sudo_modify",
    "cp": "sudo_modify",
    "rm": "sudo_delete",
    "ufw": "sudo_firewall",
    "iptables": "sudo_firewall",
    "firewall": "sudo_firewall",
}


def _categorize(command: str) -> str:
    cmd = (command or "").lower()
    for keyword, category in _COMMAND_CATEGORIES.items():
        if keyword in cmd:
            return category
    return "sudo_other"


def map_endpoint_event_to_features(event: dict) -> pd.DataFrame:
    user = event.get("user_id", "unknown")
    timestamp = event.get("timestamp")

    if timestamp:
        dt = pd.to_datetime(timestamp)
    else:
        dt = pd.Timestamp.now()

    hour = dt.hour
    day_of_week = dt.dayofweek
    is_after_hours = int(hour < 8 or hour > 18)
    is_weekend = int(day_of_week in (5, 6))

    event_name = event.get("event_name", "")
    is_firewall_change = int(event_name == "firewall_change")

    # Logstash grok extracts the command into "command"; fall back to raw message
    command = event.get("command") or event.get("message", "") or ""
    command_type = "sudo_firewall" if is_firewall_change else _categorize(command)

    key = (user, dt.date())
    _user_daily_counts[key] = _user_daily_counts.get(key, 0) + 1
    daily_endpoint_count = _user_daily_counts[key]

    return pd.DataFrame([{
        "command_type": command_type,
        "login_hour": hour,
        "day_of_week": day_of_week,
        "is_after_hours": is_after_hours,
        "is_weekend": is_weekend,
        "daily_endpoint_count": daily_endpoint_count,
        "is_firewall_change": is_firewall_change,
    }])
