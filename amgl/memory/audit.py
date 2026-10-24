"""事务内审计哈希链；可检测修改，但不能防御持有数据库的攻击者重写整条链。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

GENESIS = "0" * 64


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def append_event(conn: sqlite3.Connection, tenant: str, action: str,
                 subject: str, details: dict) -> None:
    previous = conn.execute(
        "SELECT seq, hash FROM audit WHERE tenant=? ORDER BY seq DESC LIMIT 1", (tenant,)
    ).fetchone()
    seq, prev_hash = (previous[0] + 1, previous[1]) if previous else (1, GENESIS)
    payload = canonical({"seq": seq, "tenant_id": tenant, "action": action,
                         "subject": subject, "details": details,
                         "at": datetime.now(timezone.utc).isoformat()})
    digest = hashlib.sha256((prev_hash + payload).encode()).hexdigest()
    conn.execute("INSERT INTO audit VALUES (?, ?, ?, ?, ?, ?)",
                 (tenant, seq, subject, payload, prev_hash, digest))


def verify(events: Iterable[dict]) -> bool:
    previous, expected_seq, tenant = GENESIS, 1, None
    for event in events:
        try:
            payload = event["payload"]
            data = json.loads(payload)
            tenant = tenant or data["tenant_id"]
            if data["seq"] != expected_seq or data["tenant_id"] != tenant:
                return False
            if event["prev_hash"] != previous:
                return False
            digest = hashlib.sha256((previous + payload).encode()).hexdigest()
            if event["hash"] != digest:
                return False
        except (KeyError, TypeError, ValueError):
            return False
        previous, expected_seq = digest, expected_seq + 1
    return True
