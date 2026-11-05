"""On-disk response cache. Decision models are deterministic, so a response is fully identified by
the model weights digest and the request payload."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def payload_key(model_digest: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"digest": model_digest, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                model_digest TEXT NOT NULL,
                response TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    def get(self, key: str) -> tuple[dict[str, Any], float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response, latency_ms FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), row[1]

    def put(self, key: str, model_digest: str, response: dict[str, Any], latency_ms: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                (key, model_digest, json.dumps(response, ensure_ascii=False), latency_ms, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
