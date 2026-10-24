"""AMGL 本机微基准与真实 TCP 冒烟压测；输出实测 JSON，不作生产容量背书。

突发：python -m scripts.dev.benchmark_amgl --requests 100 --concurrency 100
稳态：python -m scripts.dev.benchmark_amgl --concurrency 100 --duration-seconds 5 --warmup-seconds 2 --measurement-rounds 3
隔离：python -m scripts.dev.benchmark_amgl --server-mode isolated --http-workers 4 --concurrency 100 --duration-seconds 5
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

try:
    import resource
except ImportError:  # pragma: no cover - Windows 仅失去进程资源指标
    resource = None
try:
    import psutil
except ImportError:  # pragma: no cover - 未安装时仅缺少服务端子进程资源指标
    psutil = None

import httpx
import uvicorn

from amgl.memory.audit import verify
from amgl.memory.backends.file_kv import FileKVBackend, validate_bodies
from amgl.memory.envelope import MemoryEnvelope, Provenance, RetrievedMemory, State
from amgl.memory.read_filter import ReadPolicy, filter_results
from amgl.server import create_app


def stats(samples):
    ordered = sorted(samples)
    if not ordered:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None,
                "p99_9_ms": None, "max_ms": None, "mean_ms": None}
    def percentile(p):
        return round(ordered[max(0, math.ceil(len(ordered) * p) - 1)], 3)
    return {"samples": len(samples), "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99), "p99_9_ms": percentile(0.999),
            "max_ms": round(max(samples), 3), "mean_ms": round(statistics.mean(samples), 3)}


def resource_snapshot():
    peak_rss_bytes = None
    if resource is not None:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    return {"process_cpu_seconds": time.process_time(), "peak_rss_bytes": peak_rss_bytes}


def summarize_usage(before, after, elapsed, scope=None):
    cpu_seconds = max(0.0, after["process_cpu_seconds"] - before["process_cpu_seconds"])
    cores = os.cpu_count() or 1
    core_equivalents = cpu_seconds / elapsed if elapsed else 0.0
    rss_before, rss_after = before["peak_rss_bytes"], after["peak_rss_bytes"]
    return {
        "scope": scope or "服务端与压测客户端位于同一进程；CPU 包含两者，RSS 为进程峰值而非服务端独占值",
        "process_cpu_seconds": round(cpu_seconds, 3),
        "cpu_percent_of_one_core": round(core_equivalents * 100, 2),
        "cpu_percent_of_host_capacity": round(core_equivalents * 100 / cores, 2),
        "peak_rss_mb_before": round(rss_before / 1024 / 1024, 2) if rss_before is not None else None,
        "peak_rss_mb_after": round(rss_after / 1024 / 1024, 2) if rss_after is not None else None,
        "peak_rss_mb_growth": round(max(0, rss_after - rss_before) / 1024 / 1024, 2)
        if rss_before is not None and rss_after is not None else None,
    }


def tree_snapshot(root_pid):
    """返回进程树累计 CPU 与 RSS；采集失败交给调用方标记，不将不可见进程当成零资源。"""
    root = psutil.Process(root_pid)
    processes = [root, *root.children(recursive=True)]
    cpu, rss = {}, 0
    for process in processes:
        with process.oneshot():
            times = process.cpu_times()
            # 启动时间防止 PID 复用被误认为同一个进程。
            cpu[(process.pid, process.create_time())] = times.user + times.system
            rss += process.memory_info().rss
    return cpu, rss


def summarize_tree_usage(before_cpu, after_cpu, peak_rss, end_rss, elapsed):
    # 计量窗口内新建的进程从 0 计起；窗口内退出的进程无法计入。
    cpu_seconds = sum(max(0.0, value - before_cpu.get(pid, 0.0)) for pid, value in after_cpu.items())
    core_equivalents = cpu_seconds / elapsed if elapsed else 0.0
    return {
        "processes": len(after_cpu),
        "process_cpu_seconds": round(cpu_seconds, 3),
        "cpu_percent_of_one_core": round(core_equivalents * 100, 2),
        "cpu_percent_of_host_capacity": round(core_equivalents * 100 / (os.cpu_count() or 1), 2),
        "sampled_peak_rss_mb": round(peak_rss / 1024 / 1024, 2),
        "end_rss_mb": round(end_rss / 1024 / 1024, 2),
    }


class ProcessTreeMonitor:
    """按角色采集独立服务进程树的 CPU 与采样峰值 RSS。"""

    def __init__(self, roots, interval=0.2):
        self.roots, self.interval = roots, interval
        self._before, self._peak, self._errors = {}, {}, {}
        self._stop = threading.Event()
        self._thread = None

    def _snapshot(self, name, pid):
        try:
            cpu, rss = tree_snapshot(pid)
        except psutil.Error as error:
            self._errors[name].add(type(error).__name__)
            return {}, 0
        if name in self._before and cpu.keys() != self._before[name].keys():
            self._errors[name].add("process_tree_changed")
        return cpu, rss

    def _sample(self):
        while not self._stop.wait(self.interval):
            for name, pid in self.roots.items():
                self._peak[name] = max(self._peak[name], self._snapshot(name, pid)[1])

    def start(self):
        if psutil is None:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("资源监控已经启动")
        self._before.clear()
        for name, pid in self.roots.items():
            self._errors[name] = set()
            cpu, rss = self._snapshot(name, pid)
            self._before[name], self._peak[name] = cpu, rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def checkpoint(self):
        if psutil is None:
            return None
        result = {}
        for name, pid in self.roots.items():
            _, rss = self._snapshot(name, pid)
            result[name] = {"rss_mb": None if self._errors[name] else round(rss / 1024 / 1024, 2),
                            "sampling_errors": sorted(self._errors[name])}
        return result

    def stop(self, elapsed):
        if psutil is None:
            return {"available": False, "scope": "未安装 psutil，未采集独立 API/准入子进程资源"}
        self._stop.set()
        self._thread.join(5)
        if self._thread.is_alive():
            raise RuntimeError("资源采样线程未能退出")
        result = {"available": True, "elapsed_seconds": round(elapsed, 3),
                  "scope": ("按进程树统计（含 Uvicorn 管理/子进程）；RSS 求和含共享页，不是独占物理内存；"
                            f"峰值按 {self.interval}s 采样，可能遗漏短尖峰；CPU 窗口含读取及轮级汇总，"
                            "采样线程运行在客户端；权限失败或进程树变化的角色指标置空")}
        for name, pid in self.roots.items():
            cpu, rss = self._snapshot(name, pid)
            metrics = summarize_tree_usage(self._before[name], cpu, max(self._peak[name], rss), rss, elapsed)
            errors = sorted(self._errors[name])
            result[name] = {"complete": not errors, "sampling_errors": errors,
                            **{key: None if errors else value for key, value in metrics.items()}}
        return result


def summarize_reads(reads, elapsed, mode, target_seconds=None):
    successful = [item for item in reads if item[3] is None]
    error_counts = Counter(item[3] for item in reads if item[3] is not None)
    result = {
        "mode": mode,
        "target_seconds": target_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "completed": len(successful),
        "errors": len(reads) - len(successful),
        "error_rate": round((len(reads) - len(successful)) / len(reads), 6) if reads else 0,
        "throughput_rps": round(len(successful) / elapsed, 2) if elapsed else 0,
        **stats([item[0] for item in successful]),
        "response_bytes_mean": round(statistics.mean(item[2] for item in successful), 1) if successful else None,
        "error_types": dict(sorted(error_counts.items())),
    }
    stage_names = sorted({name for item in successful for name in item[1]})
    result["stages_ms"] = {
        name: stats([item[1][name] for item in successful if name in item[1]]) for name in stage_names
    }
    return result


class ReadAccumulator:
    """完整计数与均值，分位数采用每轮有界均匀蓄水池采样；由客户端事件循环独占。"""

    def __init__(self, sample_limit=8192):
        if not 1 <= sample_limit <= 65536:
            raise ValueError("读取采样上限必须为 1~65536")
        self.limit, self.samples = sample_limit, []
        self.completed = 0
        self.errors, self.stage_counts, self.stage_totals = Counter(), Counter(), Counter()
        self.total_latency = self.total_bytes = self.max_latency = 0
        self.stage_max = {}
        self.random = random.Random(0)

    def record(self, item):
        latency, stages, size, error = item
        if error is not None:
            self.errors[error] += 1
            return
        self.completed += 1
        self.total_latency += latency
        self.total_bytes += size
        self.max_latency = max(self.max_latency, latency)
        for name, value in stages.items():
            self.stage_counts[name] += 1
            self.stage_totals[name] += value
            self.stage_max[name] = max(self.stage_max.get(name, 0), value)
        if len(self.samples) < self.limit:
            self.samples.append(item)
        else:
            index = self.random.randrange(self.completed)
            if index < self.limit:
                self.samples[index] = item

    def summary(self, elapsed, mode, target_seconds=None):
        result = summarize_reads(self.samples, elapsed, mode, target_seconds)
        attempts = self.completed + sum(self.errors.values())
        result.update(completed=self.completed, errors=sum(self.errors.values()), samples=self.completed,
                      error_types=dict(sorted(self.errors.items())),
                      error_rate=round(sum(self.errors.values()) / attempts, 6) if attempts else 0,
                      throughput_rps=round(self.completed / elapsed, 2) if elapsed else 0,
                      max_ms=round(self.max_latency, 3) if self.completed else None,
                      mean_ms=round(self.total_latency / self.completed, 3) if self.completed else None,
                      response_bytes_mean=round(self.total_bytes / self.completed, 1) if self.completed else None,
                      quantile_samples=len(self.samples), quantile_sample_limit=self.limit,
                      quantile_scope=f"每轮最多 {self.limit} 个均匀蓄水池样本；计数/均值/max 为完整窗口；分位数是估计值")
        for name, count in self.stage_counts.items():
            stage = result["stages_ms"].setdefault(name, stats([]))
            stage.update(quantile_samples=stage["samples"], samples=count,
                         mean_ms=round(self.stage_totals[name] / count, 3),
                         max_ms=round(self.stage_max[name], 3))
        return result


def aggregate_rounds(summaries, elapsed, target_seconds):
    """按轮聚合，避免计量内存随总轮数累积；分位数取逐轮中位数。"""
    completed = sum(item["completed"] for item in summaries)
    errors = sum(item["errors"] for item in summaries)
    attempts = completed + errors
    error_types = Counter()
    for item in summaries:
        error_types.update(item["error_types"])

    def median_metric(items, key):
        values = [item[key] for item in items if item.get(key) is not None]
        return round(statistics.median(values), 3) if values else None

    def weighted_mean(items, key, weight="samples"):
        usable = [item for item in items if item.get(key) is not None and item.get(weight, 0)]
        total = sum(item[weight] for item in usable)
        return round(sum(item[key] * item[weight] for item in usable) / total, 3) if total else None

    result = {
        "mode": "sustained",
        "aggregation": "吞吐/错误率按总量计算；分位数取逐轮中位数；max 取逐轮最大值",
        "target_seconds": target_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "completed": completed,
        "errors": errors,
        "error_rate": round(errors / attempts, 6) if attempts else 0,
        "throughput_rps": round(completed / elapsed, 2) if elapsed else 0,
        "samples": completed,
        "quantile_samples": sum(item.get("quantile_samples", item["samples"]) for item in summaries),
        "quantile_scope": "分位数为每轮样本分位数的中位数；持续模式使用有界均匀采样，不是合并全量分位数",
        "p50_ms": median_metric(summaries, "p50_ms"),
        "p95_ms": median_metric(summaries, "p95_ms"),
        "p99_ms": median_metric(summaries, "p99_ms"),
        "p99_9_ms": median_metric(summaries, "p99_9_ms"),
        "max_ms": max((item["max_ms"] for item in summaries if item["max_ms"] is not None), default=None),
        "mean_ms": weighted_mean(summaries, "mean_ms"),
        "response_bytes_mean": weighted_mean(summaries, "response_bytes_mean", "completed"),
        "error_types": dict(sorted(error_types.items())),
    }
    stage_names = sorted({name for item in summaries for name in item["stages_ms"]})
    result["stages_ms"] = {}
    for name in stage_names:
        stage_items = [item["stages_ms"][name] for item in summaries if name in item["stages_ms"]]
        result["stages_ms"][name] = {
            "samples": sum(item["samples"] for item in stage_items),
            "quantile_samples": sum(item.get("quantile_samples", item["samples"]) for item in stage_items),
            "p50_ms": median_metric(stage_items, "p50_ms"),
            "p95_ms": median_metric(stage_items, "p95_ms"),
            "p99_ms": median_metric(stage_items, "p99_ms"),
            "p99_9_ms": median_metric(stage_items, "p99_9_ms"),
            "max_ms": max((item["max_ms"] for item in stage_items if item["max_ms"] is not None), default=None),
            "mean_ms": weighted_mean(stage_items, "mean_ms"),
        }
    return result


def sqlite_benchmark(backend, tenant="bench", limit=25, iterations=500):
    connection, reused, fresh, active = [], [], [], []
    for _ in range(iterations):
        start = time.perf_counter()
        with backend.connect():
            pass
        connection.append((time.perf_counter() - start) * 1000)
    with backend.connect() as conn:
        for _ in range(iterations):
            start = time.perf_counter()
            rows = conn.execute("SELECT body FROM memories WHERE tenant=? ORDER BY rowid DESC LIMIT ?",
                                (tenant, limit)).fetchall()
            validate_bodies([row[0] for row in rows])
            reused.append((time.perf_counter() - start) * 1000)
    for _ in range(iterations):
        start = time.perf_counter()
        with backend.connect() as conn:
            rows = conn.execute("SELECT body FROM memories WHERE tenant=? ORDER BY rowid DESC LIMIT ?",
                                (tenant, limit)).fetchall()
            validate_bodies([row[0] for row in rows])
        fresh.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        backend.list_memories(tenant, limit)
        active.append((time.perf_counter() - start) * 1000)
    return {"limit": limit, "connection_only": stats(connection),
            "reused_connection_read": stats(reused), "new_connection_read": stats(fresh),
            "active_backend_read": stats(active)}


def filter_benchmark(size=100, iterations=1000):
    mem = MemoryEnvelope(content="基准记忆", provenance=Provenance(session_id="bench"), tenant_id="bench")
    mem.lifecycle.state, mem.trust.score = State.COMMITTED, 0.9
    results = [RetrievedMemory(memory=mem) for _ in range(size)]
    policy = ReadPolicy()
    for _ in range(100):
        filter_results(results, policy, "bench")
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = filter_results(results, policy, "bench")
        samples.append((time.perf_counter() - start) * 1000)
        assert len(output) == size
    return {"candidates_per_call": size, **stats(samples)}


class MixedWriter:
    """单路限速提交，不积累待发任务；完整计数，延迟仅保留有界尾部样本。"""

    def __init__(self, client, rate):
        if not math.isfinite(rate) or not 0 < rate <= 100:
            raise ValueError("混合写入速率必须为有限正数且不超过 100 req/s")
        self.client, self.rate = client, rate
        self.attempts = self.completed = 0
        self.errors = Counter()
        self.unknown_outcomes = 0
        self.latencies = deque(maxlen=4096)

    async def run(self, stop):
        while not stop.is_set():
            started = time.perf_counter()
            index = self.attempts
            self.attempts += 1
            try:
                response = await self.client.post("/v1/memories:submit", json={
                    "candidate": {"content": f"混合压测独立记忆 {index}", "provenance": {"session_id": "bench"}},
                    "idempotency_key": f"mixed-{index}"})
                response.raise_for_status()
                if response.status_code != 202 or not response.json().get("id"):
                    raise ValueError("混合提交回执不符合契约")
                self.completed += 1
                self.latencies.append((time.perf_counter() - started) * 1000)
            except httpx.HTTPStatusError as error:
                self.errors[f"http_{error.response.status_code}"] += 1
            except Exception as error:
                self.errors[type(error).__name__] += 1
                # 连接中断或无效成功回执不证明未持久化；不自动重放，也不计作成功。
                self.unknown_outcomes += 1
            # 不补发错过的时间槽；过载时实际速率降低，不能冒充恒定到达率压测。
            delay = max(0.0, 1 / self.rate - (time.perf_counter() - started))
            try:
                await asyncio.wait_for(stop.wait(), delay)
            except TimeoutError:
                pass

    def summary(self, elapsed):
        return {"target_rps": self.rate, "attempts": self.attempts, "receipts": self.completed,
                "errors": sum(self.errors.values()), "error_types": dict(self.errors),
                "unknown_outcomes": self.unknown_outcomes,
                "rate_scope": "单路串行限速，不补发错过的时间槽；实际速率可能低于目标，不是恒定到达率模型",
                "receipt_rps": round(self.completed / elapsed, 2) if elapsed else 0,
                "latency_scope": "最近最多 4096 个成功提交回执；不是准入完成延迟或全窗口分位数",
                **stats(self.latencies)}


def storage_snapshot(backend, tenant="bench"):
    with backend.connect() as conn:
        conn.execute("BEGIN")
        jobs = dict(conn.execute("SELECT status, count(*) FROM jobs WHERE tenant=? GROUP BY status", (tenant,)))
        memories = conn.execute("SELECT count(*) FROM memories WHERE tenant=?", (tenant,)).fetchone()[0]
        oldest = conn.execute("SELECT min(created) FROM jobs WHERE tenant=? AND status IN ('queued','processing')",
                              (tenant,)).fetchone()[0]
    def size(path):
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
    return {"jobs": jobs, "memories": memories, "pending": jobs.get("queued", 0) + jobs.get("processing", 0),
            "oldest_pending_seconds": None if oldest is None else round(max(0, time.time() - oldest), 3),
            "db_bytes": size(backend.path), "wal_bytes": size(Path(str(backend.path) + "-wal"))}


def validate_storage(backend, expected_jobs, tenant="bench"):
    """压测停止并排空后验证计数与审计；游标流式验链，不将全部审计载入内存。"""
    with backend.connect() as conn:
        conn.execute("BEGIN")
        jobs = conn.execute("""SELECT count(*), coalesce(sum(status='accepted'),0),
                               count(DISTINCT CASE WHEN status='accepted' THEN memory_id END)
                               FROM jobs WHERE tenant=?""", (tenant,)).fetchone()
        memories = conn.execute("SELECT count(*) FROM memories WHERE tenant=?", (tenant,)).fetchone()[0]
        orphaned = conn.execute("""SELECT count(*) FROM jobs j LEFT JOIN memories m ON j.tenant=m.tenant AND j.memory_id=m.id
                                   WHERE j.tenant=? AND j.status='accepted' AND m.id IS NULL""", (tenant,)).fetchone()[0]
        actions = Counter()

        def audit_rows():
            for row in conn.execute("SELECT payload,prev_hash,hash FROM audit WHERE tenant=? ORDER BY seq", (tenant,)):
                try:
                    actions[json.loads(row["payload"])["action"]] += 1
                except (ValueError, KeyError, TypeError):
                    actions["invalid_payload"] += 1
                yield dict(row)

        audit_ok = verify(audit_rows())
        checks = {"all_receipts_persisted": jobs[0] == expected_jobs, "all_jobs_accepted": jobs[1] == expected_jobs,
                  "unique_memory_count": memories == jobs[2] == expected_jobs, "no_orphaned_receipts": orphaned == 0,
                  "audit_chain_valid": audit_ok,
                  "audit_event_counts": all(actions[action] == expected_jobs for action in
                                            ("admission.queued", "memory.created", "admission.accepted")),
                  "sqlite_quick_check": conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"}
    return {"passed": all(checks.values()), "expected_jobs": expected_jobs, "checks": checks}


async def drain_storage(backend, timeout=30):
    started = time.perf_counter()
    while True:
        state = await asyncio.to_thread(storage_snapshot, backend)
        if not state["pending"] or time.perf_counter() - started >= timeout:
            return {"drained": not state["pending"], "elapsed_seconds": round(time.perf_counter() - started, 3), **state}
        await asyncio.sleep(0.1)


async def http_benchmark(url, requests, concurrency, client_mode="isolated", read_limit=25,
                         projection="context", duration_seconds=0, warmup_seconds=0,
                         measurement_rounds=1, resource_scope=None, server_monitor=None,
                         write_rps=0, storage_backend=None):
    if not math.isfinite(write_rps) or not 0 <= write_rps <= 100 or write_rps and (not duration_seconds or requests < read_limit):
        raise ValueError("混合写入仅用于持续模式，须预填充至少 read_limit 条记忆，速率为 0~100 req/s")
    # isolated 将每个并发槽的连接池隔离，pooled 保留历史客户端负载以便对照。
    async with AsyncExitStack() as stack:
        clients = asyncio.Queue()
        count, connections = (1, concurrency) if client_mode == "pooled" else (concurrency, 1)
        for _ in range(count):
            client = await stack.enter_async_context(httpx.AsyncClient(
                base_url=url, timeout=30, trust_env=False,
                limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections)))
            clients.put_nowait(client)
        semaphore = asyncio.Semaphore(concurrency)

        @asynccontextmanager
        async def slot():
            async with semaphore:
                client = await clients.get()
                if client_mode == "pooled":
                    clients.put_nowait(client)
                try:
                    yield client
                finally:
                    if client_mode != "pooled":
                        clients.put_nowait(client)

        async def submit(i):
            async with slot() as client:
                start = time.perf_counter()
                response = await client.post("/v1/memories:submit", json={
                    "candidate": {"content": f"独立压测记忆 {i}", "provenance": {"session_id": "bench"}},
                    "idempotency_key": f"bench-{i}"})
                duration = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                return response.json()["id"], duration

        start = time.perf_counter()
        submitted = await asyncio.gather(*(submit(i) for i in range(requests)))
        submit_elapsed = time.perf_counter() - start
        pending = {ticket_id for ticket_id, _ in submitted}
        deadline = time.monotonic() + 30

        async def poll(ticket_id):
            async with slot() as client:
                response = await client.get(f"/v1/admissions/{ticket_id}")
                response.raise_for_status()
                return ticket_id, response.json()["status"]

        while pending:
            for ticket_id, status in await asyncio.gather(*(poll(i) for i in pending)):
                if status in ("failed", "rejected"):
                    raise RuntimeError("压测准入失败")
                if status == "accepted":
                    pending.remove(ticket_id)
            if time.monotonic() > deadline:
                raise TimeoutError("压测队列未在期限内清空")
            if pending:
                await asyncio.sleep(0.05)
        completion_elapsed = time.perf_counter() - start

        async def read():
            start = time.perf_counter()
            try:
                async with slot() as client:
                    trace_times = {}

                    async def trace(name, info):
                        if name == "http11.send_request_headers.started":
                            trace_times.setdefault("dispatch", time.perf_counter())
                        elif name == "http11.receive_response_headers.complete":
                            trace_times["headers"] = time.perf_counter()

                    response = await client.post("/v1/memories:filter",
                        json={"limit": read_limit, "projection": projection,
                              "policy": {"allow_provisional": True, "min_trust": 0.3}},
                        extensions={"trace": trace})
                    response.raise_for_status()
                    if len(response.json()["memories"]) != min(requests, read_limit):
                        raise ValueError("返回记忆数量不符合基准前置条件")
                    stages = {part.split(";dur=")[0].strip(): float(part.split(";dur=")[1])
                              for part in response.headers["server-timing"].split(",")}
                    stages["client_dispatch"] = (trace_times["dispatch"] - start) * 1000
                    stages["client_headers"] = (trace_times["headers"] - start) * 1000
                    return (time.perf_counter() - start) * 1000, stages, len(response.content), None
            except httpx.HTTPStatusError as error:
                return (time.perf_counter() - start) * 1000, {}, 0, f"http_{error.response.status_code}"
            except Exception as error:  # 基准必须统计错误，不能因首个异常丢失整轮结果
                return (time.perf_counter() - start) * 1000, {}, 0, type(error).__name__

        async def run_for(seconds):
            deadline = asyncio.get_running_loop().time() + seconds
            accumulator = ReadAccumulator()

            async def worker():
                while asyncio.get_running_loop().time() < deadline:
                    accumulator.record(await read())

            started = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(concurrency)))
            return accumulator.summary(time.perf_counter() - started, "sustained", seconds)

        warmup = None
        if warmup_seconds:
            warmup = await run_for(warmup_seconds)
            warmup["mode"] = "warmup"

        writer, writer_task, writer_stop = None, None, asyncio.Event()
        if write_rps:
            writer_client = await stack.enter_async_context(httpx.AsyncClient(
                base_url=url, timeout=5, trust_env=False,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1)))
            writer = MixedWriter(writer_client, write_rps)
        usage_before = resource_snapshot()
        if server_monitor is not None:
            server_monitor.start()
        measurement_started = time.perf_counter()
        try:
            initial_storage = await asyncio.to_thread(storage_snapshot, storage_backend) if storage_backend else None
            if writer:
                writer_task = asyncio.create_task(writer.run(writer_stop), name="amgl-mixed-writer")
            if duration_seconds:
                summaries, round_summaries, read_elapsed = [], [], 0.0
                for index in range(measurement_rounds):
                    summary = await run_for(duration_seconds)
                    round_elapsed = summary["elapsed_seconds"]
                    summaries.append(summary)
                    round_summaries.append({"round": index + 1, **{
                        key: summary[key] for key in ("elapsed_seconds", "completed", "errors", "error_rate",
                                                      "throughput_rps", "p50_ms", "p95_ms", "p99_ms",
                                                      "p99_9_ms", "max_ms", "mean_ms", "quantile_samples")
                    }})
                    read_elapsed += round_elapsed
                    round_summaries[-1]["storage"] = await asyncio.to_thread(storage_snapshot, storage_backend) if storage_backend else None
                    round_summaries[-1]["server_resources"] = server_monitor.checkpoint() if server_monitor else None
                    round_summaries[-1]["writer"] = {"receipts": writer.completed, "errors": sum(writer.errors.values())} if writer else None
                read_summary = aggregate_rounds(
                    summaries, read_elapsed, duration_seconds * measurement_rounds)
                read_summary["measurement_rounds"] = measurement_rounds
                read_summary["rounds"] = round_summaries
                read_summary["round_throughput_median_rps"] = round(
                    statistics.median(item["throughput_rps"] for item in round_summaries), 2)
            else:
                read_started = time.perf_counter()
                reads = await asyncio.gather(*(read() for _ in range(requests)))
                read_elapsed = time.perf_counter() - read_started
                read_summary = summarize_reads(reads, read_elapsed, "burst")
        finally:
            writer_stop.set()
            try:
                if writer_task:
                    # 等待已发出的有限请求结束，不能取消后把未知提交结果当成未发生。
                    await writer_task
            finally:
                measurement_elapsed = time.perf_counter() - measurement_started
                server_resources = server_monitor.stop(measurement_elapsed) if server_monitor is not None else None
        usage_after = resource_snapshot()
        drain = await drain_storage(storage_backend) if storage_backend else None
        # 主动检查点并观察截断：journal_size_limit 生效前，WAL 会持续增长不收缩。
        wal_after_load = None
        wal_after_checkpoint = None
        if storage_backend is not None:
            wal_after_load = storage_backend.wal_status()
            checkpoint = storage_backend.checkpoint_wal("TRUNCATE")
            wal_after_checkpoint = {**storage_backend.wal_status(), "checkpoint": checkpoint}
        validation = await asyncio.to_thread(validate_storage, storage_backend,
                                            requests + (writer.completed if writer else 0)) if storage_backend else None

        return {"requests": requests, "concurrency": concurrency, "client_mode": client_mode,
                "read_limit": read_limit, "projection": projection,
                "submit": {**stats([latency for _, latency in submitted]),
                           "batch_rps": round(requests / submit_elapsed, 2)},
                "admission_completed_seconds": round(completion_elapsed, 3),
                "warmup": warmup, "filter_http": read_summary,
                "measurement_elapsed_seconds": round(measurement_elapsed, 3),
                "measurement_resources": summarize_usage(
                    usage_before, usage_after, measurement_elapsed, resource_scope),
                "server_resources": server_resources,
                "mixed_write": writer.summary(measurement_elapsed) if writer else None,
                "storage_initial": initial_storage, "storage_drain": drain, "storage_validation": validation,
                "wal_after_load": wal_after_load, "wal_after_checkpoint": wal_after_checkpoint,
                "passed": (read_summary["errors"] == 0 and (writer is None or not writer.errors) and
                           (drain is None or drain["drained"]) and (validation is None or validation["passed"]))}


@contextmanager
def isolated_service(root, db_path, port, http_workers, read_concurrency, pool_size):
    env = os.environ.copy()
    env["AMGL_CURSOR_SECRET"] = "benchmark-shared-cursor-secret-32-bytes"
    processes = []
    command = [
        sys.executable, "-m", "amgl.server", "--db", str(db_path), "--host", "127.0.0.1",
        "--port", str(port), "--tenant", "bench", "--role", "api",
        "--http-workers", str(http_workers), "--read-concurrency", str(read_concurrency),
        "--read-pool-size", str(pool_size), "--read-timings",
    ]
    try:
        worker = subprocess.Popen([
            sys.executable, "-m", "amgl.memory.admission.worker", "--db", str(db_path),
            "--tenant", "bench", "--workers", "4", "--poll-interval", "0.01",
            "--heartbeat-interval", "1", "--heartbeat-ttl", "5", "--checkpoint-interval", "15",
        ], cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(worker)
        api = subprocess.Popen(command, cwd=root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(api)
        deadline = time.monotonic() + 15
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=1, trust_env=False) as client:
            while time.monotonic() < deadline:
                if api.poll() is not None or worker.poll() is not None:
                    raise RuntimeError("隔离压测服务提前退出")
                try:
                    # 等待就绪而非存活：确保外部准入执行体已登记心跳后再计量。
                    response = client.get("/v1/ready")
                    if response.status_code == 200 and response.json().get("role") == "api":
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError("隔离压测服务启动超时")
        yield FileKVBackend(db_path), {"api": api.pid, "admission_worker": worker.pid}
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(5)


def main():
    parser = argparse.ArgumentParser(description="AMGL 本机基准，使用临时数据库和本机临时端口")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--client-mode", choices=["pooled", "isolated"], default="isolated")
    parser.add_argument("--read-limit", type=int, default=25)
    parser.add_argument("--projection", choices=["full", "context"], default="context")
    parser.add_argument("--read-concurrency", type=int, default=2)
    parser.add_argument("--read-pool-size", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=0,
                        help="持续读取时长；0 保持固定请求数突发模式")
    parser.add_argument("--warmup-seconds", type=float, default=0,
                        help="持续模式计量前预热时长")
    parser.add_argument("--measurement-rounds", type=int, default=1,
                        help="持续模式独立计量轮数，输出逐轮及合并统计")
    parser.add_argument("--server-mode", choices=["inprocess", "isolated"], default="inprocess",
                        help="isolated 以独立准入进程和 Uvicorn 进程运行")
    parser.add_argument("--http-workers", type=int, default=1,
                        help="isolated 模式的 Uvicorn 进程数")
    parser.add_argument("--write-rps", type=float, default=0,
                        help="持续读取期间的单路限速写入；0 禁用，最大 100 req/s，不积累待发任务")
    args = parser.parse_args()
    pool_size = args.read_concurrency if args.read_pool_size is None else args.read_pool_size
    if (not 1 <= args.requests <= 10000 or not 1 <= args.concurrency <= 200 or
            not 1 <= args.read_limit <= 1000 or not 1 <= args.read_concurrency <= 32 or
            not 0 <= pool_size <= args.read_concurrency or
            not 0 <= args.duration_seconds <= 60 or not 0 <= args.warmup_seconds <= 60 or
            not 1 <= args.measurement_rounds <= 60 or not 1 <= args.http_workers <= 8 or
            args.server_mode == "inprocess" and args.http_workers != 1 or
            args.duration_seconds * args.measurement_rounds > 3600 or
            not math.isfinite(args.write_rps) or not 0 <= args.write_rps <= 100 or
            args.write_rps and (not args.duration_seconds or args.requests < args.read_limit) or
            (args.warmup_seconds or args.measurement_rounds != 1) and not args.duration_seconds):
        parser.error("请求/读取参数、持续时长或连接池大小超出范围")
    root = Path(__file__).resolve().parents[2]
    runtime_dir = root / "storage" / "amgl"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "cpu_count": os.cpu_count(), "read_concurrency": args.read_concurrency,
              "read_pool_size": pool_size, "server_mode": args.server_mode,
              "http_workers": args.http_workers,
              "scope": (("单机临时数据库；无 LLM；API/准入/压测端为独立进程；不是多机容量测试"
                         if args.server_mode == "isolated" else
                         "单机临时数据库；无 LLM；服务端与压测端同进程；不是多机容量测试")
                        + ("；持续闭环读取" if args.duration_seconds else "；短时独立提交")),
              "filter_only": [filter_benchmark(100), filter_benchmark(1000)]}
    with tempfile.TemporaryDirectory(dir=runtime_dir) as temp:
        db_path = Path(temp) / "bench.sqlite3"
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        if args.server_mode == "isolated":
            sock.close()
            with isolated_service(root, db_path, port, args.http_workers,
                                  args.read_concurrency, pool_size) as (backend, roots):
                report["real_tcp"] = asyncio.run(http_benchmark(
                    f"http://127.0.0.1:{port}", args.requests, args.concurrency,
                    args.client_mode, args.read_limit, args.projection, args.duration_seconds,
                    args.warmup_seconds, args.measurement_rounds,
                    "仅压测客户端进程；服务端见 server_resources",
                    ProcessTreeMonitor(roots), write_rps=args.write_rps, storage_backend=backend))
                report["sqlite"] = sqlite_benchmark(backend, limit=args.read_limit)
        else:
            service = create_app(db_path, tenant_id="bench", workers=4,
                                 read_timings=True, read_concurrency=args.read_concurrency,
                                 read_pool_size=pool_size)
            server = uvicorn.Server(uvicorn.Config(
                service, host="127.0.0.1", port=port, log_level="error", access_log=False))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while not server.started:
                    if not thread.is_alive() or time.monotonic() > deadline:
                        raise RuntimeError("压测服务启动失败")
                    time.sleep(0.01)
                report["real_tcp"] = asyncio.run(http_benchmark(
                    f"http://127.0.0.1:{port}", args.requests, args.concurrency,
                    args.client_mode, args.read_limit, args.projection, args.duration_seconds,
                    args.warmup_seconds, args.measurement_rounds,
                    write_rps=args.write_rps, storage_backend=service.state.backend))
                report["sqlite"] = sqlite_benchmark(service.state.backend, limit=args.read_limit)
            finally:
                server.should_exit = True
                thread.join(10)
                sock.close()
                if thread.is_alive():
                    raise RuntimeError("压测服务未能正常退出")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["real_tcp"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
