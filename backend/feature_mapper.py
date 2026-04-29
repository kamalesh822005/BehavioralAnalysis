import pandas as pd
from datetime import datetime

# simple in-memory tracking (later DB)
user_event_count = {}
user_pc_history = {}

def map_event_to_features(event: dict):
    user = event.get("user_id", "unknown")
    timestamp = event.get("timestamp")

    if timestamp:
        dt = pd.to_datetime(timestamp)
    else:
        dt = datetime.now()

    login_hour = dt.hour
    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_after_hours = 1 if login_hour < 8 or login_hour > 18 else 0

    # activity mapping
    if event.get("event_name") == "login_success":
        activity = "Logon"
    else:
        activity = "Logoff"

    pc = event.get("source_ip", "UNKNOWN-PC")

    # daily event count
    key = (user, dt.date())
    user_event_count[key] = user_event_count.get(key, 0) + 1
    daily_user_event_count = user_event_count[key]

    # new PC detection
    if user not in user_pc_history:
        user_pc_history[user] = set()

    new_pc_for_user = 0
    if pc not in user_pc_history[user]:
        new_pc_for_user = 1
        user_pc_history[user].add(pc)

    return pd.DataFrame([{
        "login_hour": login_hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_after_hours": is_after_hours,
        "daily_user_event_count": daily_user_event_count,
        "new_pc_for_user": new_pc_for_user,
        "activity": activity,
        "pc": pc
    }])