import pandas as pd
from datetime import datetime


def map_event_to_features(event: dict) -> pd.DataFrame:
    """
    Map a login/auth event to the feature vector expected by login_model.pkl.
    Features must match exactly what train_from_dataset.py uses for the login model.
    """
    timestamp = event.get("timestamp")
    dt = pd.to_datetime(timestamp) if timestamp else pd.Timestamp.now()

    hour        = dt.hour
    dow         = dt.dayofweek
    is_weekend  = int(dow in (5, 6))
    is_after    = int(hour < 8 or hour > 18)
    ssh_used    = int(bool(event.get("ssh_used")))
    event_name  = event.get("event_name", "unknown") or "unknown"

    return pd.DataFrame([{
        "event_name":     event_name,
        "login_hour":     hour,
        "day_of_week":    dow,
        "is_weekend":     is_weekend,
        "is_after_hours": is_after,
        "ssh_used":       ssh_used,
    }])
