"""文件型 KV 的 SQLite 实现：WAL、CAS、持久化租约队列与审计共用事务。

适用于单主机本地磁盘；SQLite 单写者不是分布式队列，也不支持 NFS 部署。
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from queue import Empty, LifoQueue
import sqlite3
from threading import Lock
import time

from pydantic import TypeAdapter

from amgl.errors import Conflict, NotFound, QueueFull
from ..audit import append_event, canonical
from ..envelope import AdmissionTicket, CandidateMemory, MemoryEnvelope, State, new_id
from ..admission.dedup import content_key

ENVELOPES = TypeAdapter(list[MemoryEnvelope])


def validate_bodies(bodies: list[str]) -> list[MemoryEnvelope]:
    # 数据仍逐条通过完整 schema 校验；合并到一次 core 调用，减少 Python 往返。
    return ENVELOPES.validate_json("[" + ",".join(bodies) + "]")


class FileKVBackend:
    def __init__(self, path: str | Path, *, wal_size_limit_bytes: int = 67108864):
        if str(path) == ":memory:":
            raise ValueError("持久化队列必须使用磁盘路径")
        if not 0 <= wal_size_limit_bytes <= 1 << 40:
            raise ValueError("WAL 大小上限必须为 0（不限制）到 1TiB")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wal_size_limit = wal_size_limit_bytes
        self._read_pool: LifoQueue | None = None
        self._read_pool_lock = Lock()
        self._read_pool_size = 0
        self._read_pool_closed = False
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    tenant TEXT NOT NULL, id TEXT NOT NULL, content_key TEXT NOT NULL,
                    fact_key TEXT, version INTEGER NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(tenant,id), UNIQUE(tenant,content_key)
                );
                CREATE INDEX IF NOT EXISTS memory_facts ON memories(tenant,fact_key);
                CREATE TABLE IF NOT EXISTS jobs (
                    tenant TEXT NOT NULL, id TEXT NOT NULL, idem TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, candidate TEXT, status TEXT NOT NULL,
                    memory_id TEXT, reason TEXT, owner TEXT, lease_until REAL DEFAULT 0,
                    attempts INTEGER DEFAULT 0, created REAL NOT NULL,
                    PRIMARY KEY(tenant,id), UNIQUE(tenant,idem)
                );
                CREATE INDEX IF NOT EXISTS job_claim ON jobs(tenant,status,lease_until);
                CREATE TABLE IF NOT EXISTS audit (
                    tenant TEXT NOT NULL, seq INTEGER NOT NULL, subject TEXT NOT NULL,
                    payload TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL,
                    PRIMARY KEY(tenant,seq)
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    tenant TEXT NOT NULL, run_id TEXT NOT NULL, version INTEGER NOT NULL,
                    body TEXT NOT NULL, PRIMARY KEY(tenant,run_id)
                );
                CREATE TABLE IF NOT EXISTS effects (
                    tenant TEXT NOT NULL, run_id TEXT NOT NULL, key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
                    PRIMARY KEY(tenant,run_id,key)
                );
                CREATE TABLE IF NOT EXISTS workers (
                    tenant TEXT NOT NULL, id TEXT NOT NULL, pid INTEGER NOT NULL, host TEXT NOT NULL,
                    concurrency INTEGER NOT NULL, status TEXT NOT NULL, started REAL NOT NULL,
                    heartbeat REAL NOT NULL, expires REAL NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0, last_finished REAL,
                    PRIMARY KEY(tenant,id)
                );
            """)
            # 旧库已存在 workers 表时补齐进度列，不重建表。
            columns = {row[1] for row in conn.execute("PRAGMA table_info(workers)")}
            if "completed" not in columns:
                conn.execute("ALTER TABLE workers ADD COLUMN completed INTEGER NOT NULL DEFAULT 0")
            if "last_finished" not in columns:
                conn.execute("ALTER TABLE workers ADD COLUMN last_finished REAL")

    def _new_connection(self, *, cross_thread: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5, isolation_level=None,
                               check_same_thread=not cross_thread)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        # WAL 不会因写入停顿而自动收缩；设置上限让检查点后有条件截断，防止长期占用磁盘。
        # PRAGMA 不支持参数绑定；值已在构造时校验为有限非负整数。
        if self.wal_size_limit:
            conn.execute(f"PRAGMA journal_size_limit={int(self.wal_size_limit)}")
        return conn

    @contextmanager
    def connect(self):
        conn = self._new_connection()
        try:
            yield conn
        finally:
            conn.close()

    def open_read_pool(self, size: int) -> None:
        """创建可跨工作线程、但每次只由一个线程独占的读取连接。"""
        if not 1 <= size <= 32:
            raise ValueError("读取连接池大小必须为 1~32")
        if self._read_pool is not None:
            if self._read_pool_size == size and not self._read_pool_closed:
                return
            raise RuntimeError("读取连接池已初始化")
        pool = LifoQueue(maxsize=size)
        try:
            for _ in range(size):
                pool.put_nowait(self._new_connection(cross_thread=True))
        except BaseException:
            while not pool.empty():
                pool.get_nowait().close()
            raise
        self._read_pool, self._read_pool_size, self._read_pool_closed = pool, size, False

    def close_read_pool(self) -> None:
        pool = self._read_pool
        if pool is None or self._read_pool_closed:
            return
        # 与归还操作互斥：池内连接立即关闭，已租出的连接在 finally 中自行关闭。
        with self._read_pool_lock:
            self._read_pool_closed = True
            while True:
                try:
                    pool.get_nowait().close()
                except Empty:
                    break

    @contextmanager
    def read_connection(self):
        pool = self._read_pool
        if pool is None:
            with self.connect() as conn:
                yield conn
            return
        if self._read_pool_closed:
            raise RuntimeError("读取连接池已关闭")
        try:
            conn = pool.get(timeout=5)
        except Empty:
            raise RuntimeError("读取连接池耗尽") from None
        try:
            yield conn
        finally:
            # read_many/page_memories 显式 BEGIN；归还前结束快照，避免跨请求持锁。
            try:
                if conn.in_transaction:
                    conn.rollback()
            finally:
                with self._read_pool_lock:
                    if self._read_pool_closed:
                        conn.close()
                    else:
                        pool.put_nowait(conn)

    @contextmanager
    def transaction(self):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    @staticmethod
    def _ticket(row) -> AdmissionTicket:
        return AdmissionTicket(id=row["id"], status=row["status"],
                               memory_id=row["memory_id"], reason=row["reason"])

    def submit(self, tenant: str, candidate: CandidateMemory, idem: str,
               *, rejected: str | None = None, max_queue: int = 10000) -> AdmissionTicket:
        body = candidate.model_dump(mode="json")
        # 自动生成的时间不参与幂等校验，HTTP 客户端重发相同请求仍能取得同一回执。
        identity = json.loads(canonical(body))
        identity["provenance"].pop("created_at")
        fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
        with self.transaction() as conn:
            old = conn.execute("SELECT * FROM jobs WHERE tenant=? AND idem=?", (tenant, idem)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise Conflict("幂等键已用于不同内容")
                return self._ticket(old)
            count = conn.execute("SELECT count(*) FROM jobs WHERE tenant=? AND status IN ('queued','processing')",
                                 (tenant,)).fetchone()[0]
            if not rejected and count >= max_queue:
                raise QueueFull("准入队列已满，请稍后重试")
            ticket_id = new_id()
            status = "rejected" if rejected else "queued"
            conn.execute("""INSERT INTO jobs(tenant,id,idem,fingerprint,candidate,status,reason,created)
                            VALUES (?,?,?,?,?,?,?,?)""",
                         (tenant, ticket_id, idem, fingerprint,
                          None if rejected else canonical(body), status, rejected, time.time()))
            append_event(conn, tenant, "admission." + status, ticket_id, {"reason": rejected or "queued"})
            return AdmissionTicket(id=ticket_id, status=status, reason=rejected)

    def ticket(self, tenant: str, ticket_id: str) -> AdmissionTicket:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE tenant=? AND id=?", (tenant, ticket_id)).fetchone()
        if row is None:
            raise NotFound("准入回执不存在")
        return self._ticket(row)

    def claim(self, tenant: str, lease_seconds: float) -> dict | None:
        now = time.time()
        with self.transaction() as conn:
            row = conn.execute("""SELECT * FROM jobs WHERE tenant=? AND
                                  (status='queued' OR (status='processing' AND lease_until<=?))
                                  ORDER BY created LIMIT 1""", (tenant, now)).fetchone()
            if row is None:
                return None
            owner = new_id()
            conn.execute("UPDATE jobs SET status='processing', owner=?, lease_until=?, attempts=attempts+1 WHERE tenant=? AND id=?",
                         (owner, now + lease_seconds, tenant, row["id"]))
            return {"id": row["id"], "owner": owner, "attempts": row["attempts"] + 1,
                    "candidate": CandidateMemory.model_validate_json(row["candidate"])}

    def release(self, tenant: str, job: dict, *, max_attempts: int = 3) -> None:
        status = "failed" if job["attempts"] >= max_attempts else "queued"
        with self.transaction() as conn:
            cursor = conn.execute("""UPDATE jobs SET status=?, owner=NULL, lease_until=0,
                                    reason='processing_error', candidate=CASE WHEN ?='failed' THEN NULL ELSE candidate END
                                    WHERE tenant=? AND id=? AND owner=? AND status='processing'""",
                                  (status, status, tenant, job["id"], job["owner"]))
            if cursor.rowcount:
                append_event(conn, tenant, "admission." + status, job["id"], {"reason": "processing_error"})

    def finish(self, tenant: str, job: dict, envelope: MemoryEnvelope) -> AdmissionTicket:
        if envelope.tenant_id != tenant or envelope.lifecycle.state != State.PROVISIONAL:
            raise Conflict("准入提交必须为本租户待确认记忆")
        with self.transaction() as conn:
            current = conn.execute("SELECT * FROM jobs WHERE tenant=? AND id=?", (tenant, job["id"])).fetchone()
            if (current is None or current["status"] != "processing" or
                    current["owner"] != job["owner"] or current["lease_until"] <= time.time()):
                raise Conflict("任务租约已失效")
            key = content_key(envelope.content, envelope.memory_type.value)
            duplicate = conn.execute("SELECT body FROM memories WHERE tenant=? AND content_key=?", (tenant, key)).fetchone()
            reason = envelope.governance.reason
            if duplicate:
                old = MemoryEnvelope.model_validate_json(duplicate["body"])
                memory_id, reason = old.id, "exact_duplicate"
                # 重复内容不证明真实性，不自动加信任分或复活过期记忆。
            else:
                if envelope.fact_key is not None:
                    rows = conn.execute("SELECT body FROM memories WHERE tenant=? AND fact_key=?",
                                        (tenant, envelope.fact_key)).fetchall()
                    for row in rows:
                        old = MemoryEnvelope.model_validate_json(row["body"])
                        if old.lifecycle.state in (State.PROVISIONAL, State.COMMITTED) and old.fact_value != envelope.fact_value:
                            envelope.governance.conflicts.append(old.id)
                            old.governance.conflicts.append(envelope.id)
                            old.governance.review_required = True
                            old.lifecycle.version += 1
                            conn.execute("UPDATE memories SET body=?, version=? WHERE tenant=? AND id=?",
                                         (old.model_dump_json(), old.lifecycle.version, tenant, old.id))
                            append_event(conn, tenant, "memory.conflict", old.id, {"with": envelope.id})
                memory_id = envelope.id
                conn.execute("INSERT INTO memories VALUES (?,?,?,?,?,?)",
                             (tenant, memory_id, key, envelope.fact_key, envelope.lifecycle.version, envelope.model_dump_json()))
                append_event(conn, tenant, "memory.created", memory_id,
                             {"state": envelope.lifecycle.state.value, "reason": reason})
            conn.execute("UPDATE jobs SET status='accepted', memory_id=?, reason=?, candidate=NULL, owner=NULL, lease_until=0 WHERE tenant=? AND id=?",
                         (memory_id, reason, tenant, job["id"]))
            append_event(conn, tenant, "admission.accepted", job["id"], {"memory_id": memory_id, "reason": reason})
            return AdmissionTicket(id=job["id"], status="accepted", memory_id=memory_id, reason=reason)

    def read(self, tenant: str, mem_id: str) -> MemoryEnvelope:
        with self.connect() as conn:
            row = conn.execute("SELECT body FROM memories WHERE tenant=? AND id=?", (tenant, mem_id)).fetchone()
        if row is None:
            raise NotFound("记忆不存在")
        return MemoryEnvelope.model_validate_json(row[0])

    def read_many(self, tenant: str, mem_ids: list[str], *, timings: dict | None = None) -> list[MemoryEnvelope]:
        start = time.perf_counter()
        if len(mem_ids) > 1000:
            raise ValueError("单次读取不能超过 1000 条")
        if not mem_ids:
            if timings is not None:
                timings.update(fetch=0.0, validate=0.0)
            return []
        unique_ids = list(dict.fromkeys(mem_ids))
        with self.read_connection() as conn:
            # 分批参数查询仍共用一致快照；兼容 SQLite 较低的绑定参数上限。
            conn.execute("BEGIN")
            found = {}
            for offset in range(0, len(unique_ids), 400):
                batch = unique_ids[offset:offset + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(f"SELECT id,body FROM memories WHERE tenant=? AND id IN ({placeholders})",
                                    [tenant, *batch]).fetchall()
                found.update((row["id"], row["body"]) for row in rows)
        if len(found) != len(unique_ids):
            raise NotFound("记忆不存在")
        fetched = time.perf_counter()
        # 保留调用方的顺序及重复项，不以 SQL 返回顺序代替检索顺序。
        output = validate_bodies([found[mem_id] for mem_id in mem_ids])
        if timings is not None:
            timings.update(fetch=(fetched - start) * 1000, validate=(time.perf_counter() - fetched) * 1000)
        return output

    def list_memories(self, tenant: str, limit: int = 100, *, timings: dict | None = None) -> list[MemoryEnvelope]:
        start = time.perf_counter()
        if not 1 <= limit <= 1000:
            raise ValueError("读取数量必须为 1~1000")
        with self.read_connection() as conn:
            rows = conn.execute("SELECT body FROM memories WHERE tenant=? ORDER BY rowid DESC LIMIT ?", (tenant, limit)).fetchall()
        fetched = time.perf_counter()
        output = validate_bodies([row[0] for row in rows])
        if timings is not None:
            timings.update(fetch=(fetched - start) * 1000, validate=(time.perf_counter() - fetched) * 1000)
        return output

    def page_memories(self, tenant: str, limit: int = 100, *, highwater: int | None = None,
                      before_id: str | None = None, timings: dict | None = None) -> tuple[list[MemoryEnvelope], int, str | None]:
        """以稳定 ID 倒序分页；每页一致快照，首屏 rowid 上界排除后续新插入。

        不跨请求冻结内容/状态；禁止在有效游标期间执行外部物理删除或 VACUUM。
        limit 是扫描数量，不是治理过滤后保证返回的数量。
        """
        start = time.perf_counter()
        if not 1 <= limit <= 1000 or (highwater is not None and highwater < 0):
            raise ValueError("分页边界不正确")
        with self.read_connection() as conn:
            conn.execute("BEGIN")
            if highwater is None:
                highwater = conn.execute("SELECT COALESCE(MAX(rowid),0) FROM memories WHERE tenant=?", (tenant,)).fetchone()[0]
            sql = "SELECT id,body FROM memories WHERE tenant=? AND rowid<=?"
            params = [tenant, highwater]
            if before_id is not None:
                sql += " AND id<?"
                params.append(before_id)
            sql += " ORDER BY id DESC LIMIT ?"
            rows = conn.execute(sql, [*params, limit + 1]).fetchall()
        fetched = time.perf_counter()
        scanned = rows[:limit]
        output = validate_bodies([row["body"] for row in scanned])
        if timings is not None:
            timings.update(fetch=(fetched - start) * 1000, validate=(time.perf_counter() - fetched) * 1000)
        return output, highwater, scanned[-1]["id"] if len(rows) > limit else None

    def mutate(self, tenant: str, mem_id: str, expected_version: int, mutation, *, actor: str) -> MemoryEnvelope:
        with self.transaction() as conn:
            row = conn.execute("SELECT body,version FROM memories WHERE tenant=? AND id=?", (tenant, mem_id)).fetchone()
            if row is None:
                raise NotFound("记忆不存在")
            if row["version"] != expected_version:
                raise Conflict("记忆版本冲突，请重新读取")
            mem = MemoryEnvelope.model_validate_json(row["body"])
            before = mem.lifecycle.state.value
            mutation(mem)
            mem.lifecycle.version += 1
            conn.execute("UPDATE memories SET body=?,version=? WHERE tenant=? AND id=? AND version=?",
                         (mem.model_dump_json(), mem.lifecycle.version, tenant, mem_id, expected_version))
            append_event(conn, tenant, "memory.transition", mem_id,
                         {"from": before, "to": mem.lifecycle.state.value, "actor": actor, "version": mem.lifecycle.version})
            return mem

    def events(self, tenant: str, subject: str | None = None, *, limit: int | None = None) -> list[dict]:
        with self.connect() as conn:
            sql = "SELECT payload,prev_hash,hash FROM audit WHERE tenant=?"
            params = [tenant]
            if subject is not None:
                sql += " AND subject=?"
                params.append(subject)
            sql += " ORDER BY seq"
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            return [dict(row) for row in conn.execute(sql, params)]

    def queue_depth(self, tenant: str) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT count(*) FROM jobs WHERE tenant=? AND status IN ('queued','processing')", (tenant,)).fetchone()[0]

    def queue_stats(self, tenant: str) -> tuple[int, float | None]:
        """返回未完成任务数与最早任务等待秒数；心跳只证明进程存活，积压时长反映是否在推进。"""
        with self.connect() as conn:
            depth, oldest = conn.execute("SELECT count(*), MIN(created) FROM jobs WHERE tenant=? AND status IN ('queued','processing')",
                                         (tenant,)).fetchone()
        return depth, None if oldest is None else max(0.0, time.time() - oldest)

    def heartbeat(self, tenant: str, worker_id: str, *, pid: int, host: str, concurrency: int,
                  ttl: float, status: str = "running", completed: int = 0,
                  last_finished: float | None = None) -> None:
        """准入执行体心跳属于运维状态，不写入记忆审计链；依赖同机墙钟，不跨主机使用。

        completed/last_finished 是该执行体进程生命周期内的累计值，重启后自然归零；
        进度用于停滞判断，心跳本身只证明进程存活。
        """
        if (status not in ("running", "stopping") or not math.isfinite(ttl) or ttl <= 0 or
                not math.isfinite(completed) or completed < 0 or
                last_finished is not None and not math.isfinite(last_finished)):
            raise ValueError("心跳状态或有效期不正确")
        now = time.time()
        with self.transaction() as conn:
            conn.execute("""INSERT INTO workers(tenant,id,pid,host,concurrency,status,started,heartbeat,
                                                expires,completed,last_finished)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(tenant,id) DO UPDATE SET status=excluded.status,
                            heartbeat=excluded.heartbeat, expires=excluded.expires,
                            completed=excluded.completed, last_finished=excluded.last_finished""",
                         (tenant, worker_id, pid, host, concurrency, status, now, now, now + ttl,
                          int(completed), last_finished))
            # 异常退出的执行体无法自行注销；尾随清理超时较久的记录，避免表无界增长。
            conn.execute("DELETE FROM workers WHERE tenant=? AND expires<?", (tenant, now - 3600))

    def deregister(self, tenant: str, worker_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM workers WHERE tenant=? AND id=?", (tenant, worker_id))

    def admission_workers(self, tenant: str) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute("SELECT status, count(*) FROM workers WHERE tenant=? AND expires>? GROUP BY status",
                                (tenant, time.time())).fetchall()
        counts = {row[0]: row[1] for row in rows}
        return {"ready": counts.get("running", 0), "draining": counts.get("stopping", 0)}

    def admission_progress(self, tenant: str) -> dict:
        """聚合在线执行体的完成进度；任一执行体最近完成即视为仍在推进。"""
        now = time.time()
        with self.connect() as conn:
            ready, completed, last = conn.execute(
                """SELECT count(*), coalesce(sum(completed),0), max(last_finished) FROM workers
                   WHERE tenant=? AND status='running' AND expires>?""", (tenant, now)).fetchone()
        return {"ready_workers": ready, "completed_total": completed,
                "last_finished_seconds_ago": None if last is None else round(max(0.0, now - last), 3)}

    def checkpoint_wal(self, mode: str = "PASSIVE") -> dict:
        """主动推进 WAL 检查点；PASSIVE 不阻塞并发读写，截断由 journal_size_limit 决定。"""
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            raise ValueError("WAL 检查点模式不正确")
        with self.connect() as conn:
            busy, log_pages, checkpointed = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return {"mode": mode, "busy": bool(busy), "wal_pages": log_pages,
                "checkpointed_pages": checkpointed}

    def wal_status(self) -> dict:
        """journal_size_limit 为 SQLite 原值：-1 表示不限制，其他为字节上限。"""
        with self.connect() as conn:
            limit = conn.execute("PRAGMA journal_size_limit").fetchone()[0]
        try:
            wal_bytes = Path(str(self.path) + "-wal").stat().st_size
        except FileNotFoundError:
            wal_bytes = 0
        return {"wal_bytes": wal_bytes, "journal_size_limit_bytes": limit}
