"""安全恢复：本地事务能去重已完成结果，不能保证外部副作用 exactly-once。"""
from __future__ import annotations

import asyncio
import hashlib
import json

from amgl.errors import Conflict, NotFound, UnsafeResume
from amgl.memory.audit import append_event, canonical
from amgl.memory.backends import FileKVBackend


def idempotency_key(tenant: str, run_id: str, step: str) -> str:
    return hashlib.sha256(canonical([tenant, run_id, step]).encode()).hexdigest()


class RecoveryStore:
    def __init__(self, backend: FileKVBackend, tenant: str = "local"):
        self.backend, self.tenant = backend, tenant

    def snapshot(self, run_id: str, state: dict, *, expected_version: int = 0) -> int:
        payload = canonical(state)
        with self.backend.transaction() as conn:
            old = conn.execute("SELECT version FROM snapshots WHERE tenant=? AND run_id=?", (self.tenant, run_id)).fetchone()
            version = old[0] if old else 0
            if version != expected_version:
                raise Conflict("运行检查点版本冲突")
            conn.execute("INSERT INTO snapshots VALUES (?,?,?,?) ON CONFLICT(tenant,run_id) DO UPDATE SET version=excluded.version,body=excluded.body",
                         (self.tenant, run_id, version + 1, payload))
            append_event(conn, self.tenant, "loop.snapshot", run_id, {"version": version + 1})
            return version + 1

    def restore(self, run_id: str) -> tuple[dict, int]:
        with self.backend.connect() as conn:
            old = conn.execute("SELECT body,version FROM snapshots WHERE tenant=? AND run_id=?", (self.tenant, run_id)).fetchone()
        if old is None:
            raise NotFound("运行检查点不存在")
        return json.loads(old[0]), old[1]

    def _reserve(self, run_id: str, step: str, inputs) -> tuple[bool, object]:
        key = idempotency_key(self.tenant, run_id, step)
        fingerprint = hashlib.sha256(canonical(inputs).encode()).hexdigest()
        with self.backend.transaction() as conn:
            old = conn.execute("SELECT * FROM effects WHERE tenant=? AND run_id=? AND key=?", (self.tenant, run_id, key)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise Conflict("工具幂等键对应的输入已改变")
                if old["status"] == "done":
                    return True, json.loads(old["result"])
                raise UnsafeResume("工具结果未知，必须查询外部结果并核对后才能恢复")
            conn.execute("INSERT INTO effects VALUES (?,?,?,?,?,NULL)",
                         (self.tenant, run_id, key, fingerprint, "pending"))
            append_event(conn, self.tenant, "loop.intent", run_id, {"key": key})
        return False, key

    def reconcile(self, run_id: str, step: str, result, *, actor: str) -> None:
        """仅由可信外部核对器/人工调用；不是根据模型猜测补写结果。"""
        key = idempotency_key(self.tenant, run_id, step)
        payload = canonical(result)
        with self.backend.transaction() as conn:
            cursor = conn.execute("UPDATE effects SET status='done',result=? WHERE tenant=? AND run_id=? AND key=? AND status='pending'",
                                  (payload, self.tenant, run_id, key))
            if cursor.rowcount != 1:
                raise Conflict("工具意图不存在或已经完成")
            append_event(conn, self.tenant, "loop.completed", run_id, {"key": key, "actor": actor})

    async def execute(self, run_id: str, step: str, inputs, operation):
        """operation 为 async 函数，接收稳定幂等键，结果必须可 JSON 序列化。

        若调用失败/取消，保留 pending，即使异常是断网，也不能断定远端没执行。
        """
        cached, value = await asyncio.to_thread(self._reserve, run_id, step, inputs)
        if cached:
            return value
        result = await operation(value)
        await asyncio.to_thread(self.reconcile, run_id, step, result, actor="executor")
        return result
