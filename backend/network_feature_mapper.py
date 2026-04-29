import pandas as pd
from datetime import datetime

def map_network_event_to_features(event: dict) -> pd.DataFrame:
    timestamp = event.get("timestamp")
    if timestamp:
        dt = pd.to_datetime(timestamp)
    else:
        dt = pd.Timestamp.now()

    event_type = event.get("event_name", "flow") or "flow"
    proto = event.get("proto", "unknown") or "unknown"
    app_proto = event.get("app_proto", "unknown") or "unknown"

    src_port = int(event.get("src_port", 0) or 0)
    dest_port = int(event.get("dest_port", 0) or 0)
    bytes_toserver = float(event.get("bytes_toserver", 0) or 0)
    bytes_toclient = float(event.get("bytes_toclient", 0) or 0)
    flow_age = float(event.get("flow_age", 0) or 0)
    alert_severity = float(event.get("alert_severity", 0) or 0)

    hour = dt.hour
    day_of_week = dt.dayofweek

    return pd.DataFrame([{
        "event_type": event_type,
        "proto": proto,
        "app_proto": app_proto,
        "src_port": src_port,
        "dest_port": dest_port,
        "bytes_toserver": bytes_toserver,
        "bytes_toclient": bytes_toclient,
        "flow_age": flow_age,
        "alert_severity": alert_severity,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_alert": int(event_type == "alert"),
        "is_dns": int(event_type == "dns"),
        "is_http": int(event_type == "http"),
        "is_tls": int(event_type == "tls"),
        "is_flow": int(event_type == "flow"),
    }])
