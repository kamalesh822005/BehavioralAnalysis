"""
SQLite persistence layer for BLATS.

Tables:
  events            - every processed event (login, network, endpoint)
  users             - current trust state per user (upserted on each event)
  trust_history     - time-series of trust score changes per user
"""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime

DB_PATH = Path(r"C:\BLATS\dataset\blats.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection."""
    if not getattr(_local, "conn", None):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            user_id          TEXT,
            event_name       TEXT,
            category         TEXT,
            severity         TEXT,
            rule_risk        INTEGER,
            ml_login_risk    INTEGER,
            ml_network_risk  INTEGER,
            ml_endpoint_risk INTEGER,
            final_risk       INTEGER,
            trust_score      INTEGER,
            source_ip        TEXT,
            frontend_message TEXT,
            raw_message      TEXT,
            created_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            trust_score INTEGER DEFAULT 100,
            status      TEXT    DEFAULT 'trusted',
            event_count INTEGER DEFAULT 0,
            last_event  TEXT,
            last_risk   INTEGER DEFAULT 0,
            updated_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS trust_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT,
            trust_score INTEGER,
            status      TEXT,
            final_risk  INTEGER,
            event_name  TEXT,
            timestamp   TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_events_category  ON events(category);
        CREATE INDEX IF NOT EXISTS idx_events_user      ON events(user_id);
        CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trust_user       ON trust_history(user_id);
    """)
    c.commit()


# ── Write ─────────────────────────────────────────────────────────────────────

def save_event(ev: dict):
    c = _conn()
    c.execute("""
        INSERT INTO events
          (timestamp, user_id, event_name, category, severity,
           rule_risk, ml_login_risk, ml_network_risk, ml_endpoint_risk,
           final_risk, trust_score, source_ip, frontend_message, raw_message)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        ev.get("timestamp"),
        ev.get("user_id"),
        ev.get("event_name"),
        ev.get("category"),
        ev.get("severity"),
        ev.get("rule_risk"),
        ev.get("ml_login_risk"),
        ev.get("ml_network_risk"),
        ev.get("ml_endpoint_risk"),
        ev.get("final_risk"),
        ev.get("trust_score"),
        ev.get("source_ip"),
        ev.get("frontend_message"),
        ev.get("raw_message"),
    ))
    c.commit()


def save_user(user_id: str, state: dict):
    c = _conn()
    c.execute("""
        INSERT INTO users (user_id, trust_score, status, event_count, last_event, last_risk, updated_at)
        VALUES (?,?,?,?,?,?, datetime('now'))
        ON CONFLICT(user_id) DO UPDATE SET
            trust_score = excluded.trust_score,
            status      = excluded.status,
            event_count = excluded.event_count,
            last_event  = excluded.last_event,
            last_risk   = excluded.last_risk,
            updated_at  = datetime('now')
    """, (
        user_id,
        state.get("trust_score", 100),
        state.get("status", "trusted"),
        state.get("event_count", 0),
        state.get("last_event"),
        state.get("last_risk", 0),
    ))
    c.commit()


def save_trust_history(user_id: str, trust_score: int, status: str,
                       final_risk: int, event_name: str):
    c = _conn()
    c.execute("""
        INSERT INTO trust_history (user_id, trust_score, status, final_risk, event_name)
        VALUES (?,?,?,?,?)
    """, (user_id, trust_score, status, final_risk, event_name))
    c.commit()


# ── Read ──────────────────────────────────────────────────────────────────────

def load_users() -> dict:
    """Load all users from DB into the in-memory dict format."""
    rows = _conn().execute("SELECT * FROM users").fetchall()
    return {r["user_id"]: {
        "trust_score": r["trust_score"],
        "status":      r["status"],
        "event_count": r["event_count"],
        "last_event":  r["last_event"],
        "last_risk":   r["last_risk"],
    } for r in rows}


def get_events_by_category(category: str, limit: int = 200) -> list:
    rows = _conn().execute(
        "SELECT * FROM events WHERE category=? ORDER BY id DESC LIMIT ?",
        (category, limit)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_all_events(limit: int = 200) -> list:
    rows = _conn().execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_trust_history(user_id: str, limit: int = 100) -> list:
    rows = _conn().execute(
        "SELECT * FROM trust_history WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_stats() -> dict:
    c = _conn()
    total      = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    login      = c.execute("SELECT COUNT(*) FROM events WHERE category='login'").fetchone()[0]
    network    = c.execute("SELECT COUNT(*) FROM events WHERE category='network'").fetchone()[0]
    endpoint   = c.execute("SELECT COUNT(*) FROM events WHERE category='endpoint'").fetchone()[0]
    high_risk  = c.execute("SELECT COUNT(*) FROM events WHERE final_risk>=60").fetchone()[0]
    users_cnt  = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    risky_cnt  = c.execute("SELECT COUNT(*) FROM users WHERE status='risky'").fetchone()[0]
    watch_cnt  = c.execute("SELECT COUNT(*) FROM users WHERE status='watch'").fetchone()[0]

    # event counts per hour for the last 24h (login + network separately)
    login_hourly = c.execute("""
        SELECT strftime('%H', timestamp) as hr, COUNT(*) as cnt
        FROM events WHERE category='login'
          AND timestamp >= datetime('now','-1 day')
        GROUP BY hr ORDER BY hr
    """).fetchall()

    net_hourly = c.execute("""
        SELECT strftime('%H', timestamp) as hr, COUNT(*) as cnt
        FROM events WHERE category='network'
          AND timestamp >= datetime('now','-1 day')
        GROUP BY hr ORDER BY hr
    """).fetchall()

    return {
        "total_events":    total,
        "login_events":    login,
        "network_events":  network,
        "endpoint_events": endpoint,
        "high_risk_events": high_risk,
        "total_users":     users_cnt,
        "risky_users":     risky_cnt,
        "watch_users":     watch_cnt,
        "login_hourly":    [{"hour": r["hr"], "count": r["cnt"]} for r in login_hourly],
        "network_hourly":  [{"hour": r["hr"], "count": r["cnt"]} for r in net_hourly],
    }
