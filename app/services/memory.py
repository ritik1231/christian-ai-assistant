"""
SQLite-backed conversation memory.
Stores per-session message history and denomination preference.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from app.config import BASE_DIR, HISTORY_TURNS

DB_PATH = BASE_DIR / "data" / "memory.db"


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they don't exist. Call once at app startup."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                denomination TEXT NOT NULL DEFAULT 'Non-denominational (default)',
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)


def new_session(denomination: str = "Non-denominational (default)") -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (id, denomination) VALUES (?, ?)",
            (session_id, denomination),
        )
    return session_id


def get_or_create_session(
    session_id: str | None,
    denomination: str = "Non-denominational (default)",
) -> str:
    """Return existing session_id or create a new one."""
    if session_id:
        with _conn() as con:
            row = con.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row:
            return session_id
    return new_session(denomination)


def update_denomination(session_id: str, denomination: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE sessions SET denomination = ? WHERE id = ?",
            (denomination, session_id),
        )


def get_denomination(session_id: str) -> str:
    with _conn() as con:
        row = con.execute(
            "SELECT denomination FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return row["denomination"] if row else "Non-denominational (default)"


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session history. role: 'user' | 'assistant'"""
    with _conn() as con:
        con.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_history(session_id: str, last_n: int = HISTORY_TURNS) -> list[dict]:
    """Return the last `last_n` turns as [{role, content}, ...] oldest-first."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, last_n),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_session(session_id: str) -> None:
    """Delete all messages for a session (keeps session row)."""
    with _conn() as con:
        con.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
