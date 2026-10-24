"""FastAPI 外壳，仅调用组件；默认单主机单租户，不开放任意工具执行。"""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import math
import os
from typing import Literal
import sqlite3
import time

from fastapi import Depends, FastAPI, Header, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import Field, TypeAdapter
from starlette.exceptions import HTTPException

from .errors import AMGLError, NotReady
from .loop.recovery import RecoveryStore
from .memory.admission.pipeline import AdmissionPipeline
from .memory.backends import FileKVBackend
from .memory.audit import canonical
from .memory.envelope import CandidateMemory, FilteredMemory, Model, RetrievedMemory, State
from .memory.lifecycle import LifecycleManager
from .memory.read_filter import ReadPolicy, filter_results


class SubmitRequest(Model):
    candidate: CandidateMemory
    idempotency_key: str = Field(min_length=1, max_length=128)


class FilterRequest(Model):
    memory_ids: list[str] = Field(default_factory=list, max_length=1000)
    policy: ReadPolicy = Field(default_factory=ReadPolicy)
    limit: int = Field(default=100, ge=1, le=1000)
    projection: Literal["full", "context"] = "full"


class PageRequest(Model):
    policy: ReadPolicy = Field(default_factory=ReadPolicy)
    limit: int = Field(default=25, ge=1, le=1000)
    projection: Literal["full", "context"] = "context"
    cursor: str | None = Field(default=None, min_length=1, max_length=1024)


class PagePosition(Model):
    v: Literal[1] = 1
    highwater: int = Field(ge=0)
    before_id: str = Field(min_length=1, max_length=128)
    scope: str
    expires: int


FILTERED_LIST = TypeAdapter(list[FilteredMemory])
CONTEXT_INCLUDE = {"__all__": {
    "memory": {"id": True, "content": True, "memory_type": True, "expires_at": True,
               "lifecycle": {"state", "version"}, "trust": {"score"},
               "provenance": {"source_tier"}},
    "score": True, "annotations": True,
}}


class TransitionRequest(Model):
    memory_id: str = Field(min_length=1, max_length=128)
    to_state: State
    expected_version: int = Field(ge=1)
    reviewed: bool = False


class BodyLimitMiddleware:
    """在解析 JSON 前限制整个请求体，包含没有 Content-Length 的请求。"""
    def __init__(self, app, limit: int = 65536):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10
        while True:
            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    raise TimeoutError
                message = await asyncio.wait_for(receive(), remaining)
            except TimeoutError:
                return await JSONResponse({"code": 40801, "message": "请求体接收超时"}, status_code=408)(scope, receive, send)
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                return await JSONResponse({"code": 41301, "message": "请求体超过大小限制"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(db_path="storage/amgl/amgl.sqlite3", *, tenant_id="local",
               api_token: str | None = None, admin_token: str | None = None,
               judge=None, workers: int = 2, max_queue: int = 10000, read_concurrency: int = 2,
               read_pool_size: int | None = None, read_timings: bool = False,
               cursor_secret: bytes | None = None, role: Literal["all", "api"] = "all",
               wal_size_limit_bytes: int = 67108864,
               stall_stale_seconds: float = 120) -> FastAPI:
    pool_size = read_concurrency if read_pool_size is None else read_pool_size
    if role not in ("all", "api"):
        raise ValueError("HTTP 进程角色必须为 all 或 api")
    if not 1 <= read_concurrency <= 32 or not 0 <= pool_size <= read_concurrency:
        raise ValueError("数据库读并发必须为 1~32，连接池大小必须为 0~读并发")
    if cursor_secret is not None and len(cursor_secret) < 32:
        raise ValueError("游标签名密钥至少为 32 字节")
    if not math.isfinite(stall_stale_seconds) or stall_stale_seconds <= 0:
        raise ValueError("停滞告警阈值必须为正有限数")
    signing_key = cursor_secret if cursor_secret is not None else os.urandom(32)
    read_slots = asyncio.Semaphore(read_concurrency)
    backend = FileKVBackend(db_path, wal_size_limit_bytes=wal_size_limit_bytes)
    pipeline = AdmissionPipeline(backend, tenant_id, judge=judge, workers=workers, max_queue=max_queue)
    lifecycle = LifecycleManager(backend)

    @asynccontextmanager
    async def lifespan(app):
        if pool_size:
            backend.open_read_pool(pool_size)
        if role == "all":
            await pipeline.start()
        try:
            yield
        finally:
            if role == "all":
                await pipeline.stop()
            backend.close_read_pool()

    app = FastAPI(title="AMGL", version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimitMiddleware)
    app.state.backend, app.state.pipeline, app.state.role = backend, pipeline, role

    def matches(header, token):
        return bool(token and header and hmac.compare_digest(header.encode(), ("Bearer " + token).encode()))

    async def authenticated(authorization: str | None = Header(default=None)):
        if api_token and not (matches(authorization, api_token) or matches(authorization, admin_token)):
            raise HTTPException(401, "身份验证失败")

    async def admin(authorization: str | None = Header(default=None)):
        if not matches(authorization, admin_token):
            raise HTTPException(403, "此操作需要管理员权限")

    @app.exception_handler(AMGLError)
    async def domain_error(request, error):
        return JSONResponse({"code": error.code, "message": str(error)}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # 不回显 error.errors()，其中可能包含密钥原文。
        return JSONResponse({"code": 42201, "message": "请求参数格式不正确"}, status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({"code": error.status_code * 100 + 1, "message": str(error.detail)}, status_code=error.status_code)

    @app.exception_handler(sqlite3.Error)
    async def database_error(request, error):
        return JSONResponse({"code": 50301, "message": "持久化存储暂不可用"}, status_code=503)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        return JSONResponse({"code": 50001, "message": "服务内部错误"}, status_code=500)

    @app.post("/v1/memories:submit", status_code=202, dependencies=[Depends(authenticated)])
    async def submit(request: SubmitRequest):
        return await pipeline.submit(request.candidate, request.idempotency_key)

    @app.get("/v1/admissions/{ticket_id}", dependencies=[Depends(authenticated)])
    async def ticket(ticket_id: str):
        return await asyncio.to_thread(backend.ticket, tenant_id, ticket_id)

    def check_policy(policy, authorization):
        if (policy.allow_pii or policy.allow_conflicts) and not matches(authorization, admin_token):
            raise HTTPException(403, "读取敏感或冲突记忆需要管理员权限")

    def cursor_scope(request):
        return hashlib.sha256(canonical([str(backend.path), tenant_id, request.policy.model_dump(),
                                         request.projection, request.limit]).encode()).hexdigest()

    def encode_cursor(position):
        payload = base64.urlsafe_b64encode(position.model_dump_json().encode()).rstrip(b"=")
        signature = hmac.new(signing_key, payload, hashlib.sha256).hexdigest().encode()
        return (payload + b"." + signature).decode()

    def decode_cursor(token, scope):
        try:
            payload, signature = token.encode("ascii").split(b".")
            expected = hmac.new(signing_key, payload, hashlib.sha256).hexdigest().encode()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            decoded = base64.b64decode(payload + b"=" * (-len(payload) % 4), altchars=b"-_", validate=True)
            position = PagePosition.model_validate_json(decoded)
            if position.scope != scope or position.expires <= time.time():
                raise ValueError
            return position
        except (ValueError, binascii.Error):
            raise HTTPException(400, "游标无效、过期或不匹配当前读取条件") from None

    async def filtered_response(request, retrieve):
        timings = {}
        queued = time.perf_counter()
        def work(acquired):
            started = time.perf_counter()
            memories, metadata = retrieve(timings)
            fetched = time.perf_counter()
            results = filter_results([RetrievedMemory(memory=m) for m in memories], request.policy, tenant_id)
            filtered = time.perf_counter()
            # 先完成全部治理检查，再应用固定白名单投影；不返回未经验证的数据库原文。
            include = CONTEXT_INCLUDE if request.projection == "context" else None
            body = b'{"memories":' + FILTERED_LIST.dump_json(results, include=include)
            if metadata is not None:
                body += b"," + json.dumps(metadata, separators=(",", ":")).encode()[1:-1]
            body += b"}"
            timings.update(dispatch=(started - acquired) * 1000, filter=(filtered - fetched) * 1000,
                           serialize=(time.perf_counter() - filtered) * 1000)
            return body
        # 整个读/过滤/编码执行体占用同一个有界槽，不能放大到无限编码线程。
        async with read_slots:
            acquired = time.perf_counter()
            task = asyncio.create_task(asyncio.to_thread(work, acquired))
            try:
                body = await asyncio.shield(task)
            except asyncio.CancelledError:
                # 线程不能强制取消；实际结束前不释放槽，防止取消请求绕过并发限制。
                await asyncio.gather(task, return_exceptions=True)
                raise
        finished = time.perf_counter()
        timings.update(queue=(acquired - queued) * 1000, read=(finished - acquired) * 1000,
                       service=(finished - queued) * 1000)
        headers = {"Cache-Control": "no-store"}
        if read_timings:
            headers["Server-Timing"] = ", ".join(f"{name};dur={value:.3f}" for name, value in timings.items())
        return Response(body, media_type="application/json", headers=headers)

    @app.post("/v1/memories:filter", dependencies=[Depends(authenticated)])
    async def read(request: FilterRequest, authorization: str | None = Header(default=None)):
        check_policy(request.policy, authorization)
        def retrieve(timings):
            if request.memory_ids:
                memories = backend.read_many(tenant_id, request.memory_ids, timings=timings)
            else:
                memories = backend.list_memories(tenant_id, request.limit, timings=timings)
            return memories, None
        return await filtered_response(request, retrieve)

    @app.post("/v1/memories:page", dependencies=[Depends(authenticated)])
    async def page(request: PageRequest, authorization: str | None = Header(default=None)):
        check_policy(request.policy, authorization)
        scope = cursor_scope(request)
        position = decode_cursor(request.cursor, scope) if request.cursor else None
        expires = position.expires if position else int(time.time()) + 300
        def retrieve(timings):
            # 排队期间也可能过期；读取时再次核对，不因翻页续期。
            if position and position.expires <= time.time():
                raise HTTPException(400, "游标已过期")
            memories, highwater, before_id = backend.page_memories(
                tenant_id, request.limit, highwater=position.highwater if position else None,
                before_id=position.before_id if position else None, timings=timings)
            next_cursor = encode_cursor(PagePosition(highwater=highwater, before_id=before_id,
                                        scope=scope, expires=expires)) if before_id else None
            return memories, {"next_cursor": next_cursor, "scanned": len(memories)}
        return await filtered_response(request, retrieve)

    @app.get("/v1/memories/{mem_id}", dependencies=[Depends(admin)])
    async def raw_memory(mem_id: str):
        return await asyncio.to_thread(backend.read, tenant_id, mem_id)

    @app.post("/v1/lifecycle:transition", dependencies=[Depends(admin)])
    async def transition(request: TransitionRequest):
        return await asyncio.to_thread(lifecycle.transition, tenant_id, request.memory_id, request.to_state,
                                       request.expected_version, actor="admin", reviewed=request.reviewed)

    @app.get("/v1/audit:query", dependencies=[Depends(admin)])
    async def audit(mem_id: str | None = None, limit: int = Query(default=100, ge=1, le=1000)):
        return {"events": await asyncio.to_thread(backend.events, tenant_id, mem_id, limit=limit)}

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(admin)])
    async def run_state(run_id: str):
        state, version = await asyncio.to_thread(RecoveryStore(backend, tenant_id).restore, run_id)
        return {"state": state, "version": version}

    def component_status(local_ready: bool):
        # 本地执行体状态在事件循环内读取，线程内只访问存储。
        workers_seen = None if role == "all" else backend.admission_workers(tenant_id)
        ready = local_ready if role == "all" else workers_seen["ready"] > 0
        progress = pipeline.progress() if role == "all" else backend.admission_progress(tenant_id)
        depth, oldest = backend.queue_stats(tenant_id)
        last_finished = progress["last_finished_seconds_ago"]
        # 心跳在线但积压长期不消化：队列有任务、最近完成缺失或超过阈值。
        stalled = bool(depth and ready and
                       (last_finished is None or last_finished > stall_stale_seconds))
        return {"role": role, "admission_execution": "local" if role == "all" else "external",
                "admission_ready": ready, "admission_workers": workers_seen,
                "admission_progress": progress, "admission_stalled": stalled,
                "queued": depth,
                "oldest_pending_seconds": None if oldest is None else round(oldest, 3)}

    @app.get("/v1/health", dependencies=[Depends(authenticated)])
    async def health():
        # 存活探针：api 角色即使暂无外部执行体仍可服务读取与持久化提交，不应被重启。
        if role == "all" and not pipeline.healthy:
            raise NotReady("准入执行体尚未就绪")
        status = await asyncio.to_thread(component_status, pipeline.healthy)
        return {"status": "ok", **status, "judge_configured": judge is not None,
                "read_concurrency": read_concurrency, "read_pool_size": pool_size,
                "wal": backend.wal_status()}

    @app.get("/v1/ready", dependencies=[Depends(authenticated)])
    async def ready():
        # 就绪探针：存储可读且至少一个本地或未过期的外部准入执行体在线；
        # 停滞是降级信号而非不可用，返回 200 附带 admission_stalled，由调用方决定是否摘流量。
        status = await asyncio.to_thread(component_status, pipeline.healthy)
        if not status["admission_ready"]:
            raise NotReady("准入执行体未就绪或心跳已过期")
        return {"status": "ready", **status}

    return app


def _runtime_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} 必须为整数") from error
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} 超出范围")
    return value


def _runtime_float(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} 必须为数字") from error
    if not math.isfinite(value) or value < minimum:
        raise RuntimeError(f"{name} 超出范围")
    return value


def create_app_from_env() -> FastAPI:
    """供 Uvicorn spawn worker 导入；只读取父进程写入的 AMGL_RUNTIME_* 配置。"""
    role = os.getenv("AMGL_RUNTIME_ROLE", "api")
    pool_raw = os.getenv("AMGL_RUNTIME_READ_POOL_SIZE", "")
    pool_size = None if pool_raw == "" else _runtime_int("AMGL_RUNTIME_READ_POOL_SIZE", 0, 0, 32)
    secret = os.getenv("AMGL_CURSOR_SECRET")
    wal_limit = _runtime_int("AMGL_RUNTIME_WAL_LIMIT", 67108864, 0, 1 << 40)
    return create_app(
        os.getenv("AMGL_RUNTIME_DB", "storage/amgl/amgl.sqlite3"),
        tenant_id=os.getenv("AMGL_RUNTIME_TENANT", "local"),
        api_token=os.getenv("AMGL_API_TOKEN"), admin_token=os.getenv("AMGL_ADMIN_TOKEN"),
        workers=_runtime_int("AMGL_RUNTIME_ADMISSION_WORKERS", 2, 1, 32),
        max_queue=_runtime_int("AMGL_RUNTIME_MAX_QUEUE", 10000, 1, 1000000),
        read_concurrency=_runtime_int("AMGL_RUNTIME_READ_CONCURRENCY", 2, 1, 32),
        read_pool_size=pool_size, read_timings=os.getenv("AMGL_RUNTIME_READ_TIMINGS") == "1",
        cursor_secret=secret.encode() if secret is not None else None, role=role,
        wal_size_limit_bytes=wal_limit,
        stall_stale_seconds=_runtime_float("AMGL_RUNTIME_STALL_STALE_SECONDS", 120, 0.001))


def main():
    parser = argparse.ArgumentParser(description="AMGL 本地记忆与 Loop 组件服务")
    parser.add_argument("--db", default="storage/amgl/amgl.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--role", choices=["all", "api"], default="all",
                        help="all 为兼容单进程；api 不启动准入执行体，可安全扩展 HTTP 进程")
    parser.add_argument("--workers", default=2, type=int, help="all 角色的准入协程数量")
    parser.add_argument("--http-workers", default=1, type=int, help="Uvicorn 进程数；大于 1 仅允许 api 角色")
    parser.add_argument("--max-queue", default=10000, type=int)
    parser.add_argument("--read-timings", action="store_true", help="在读响应中启用 Server-Timing 诊断头")
    parser.add_argument("--read-concurrency", type=int, default=2, help="每个 HTTP 进程的读取执行体并发上限")
    parser.add_argument("--read-pool-size", type=int, default=None, help="每个 HTTP 进程的 SQLite 读连接池；默认等于读并发，0 为禁用")
    parser.add_argument("--wal-size-limit", type=int, default=67108864,
                        help="WAL 文件大小上限（字节），0 为不限制；检查点后有条件截断")
    parser.add_argument("--stall-stale-seconds", type=float, default=120,
                        help="队列积压且最近完成超过该秒数时标记 admission_stalled")
    args = parser.parse_args()
    token, admin_token = os.getenv("AMGL_API_TOKEN"), os.getenv("AMGL_ADMIN_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        parser.error("非本机监听必须配置 AMGL_API_TOKEN，并在网关配置 TLS 和限流")
    secret = os.getenv("AMGL_CURSOR_SECRET")
    if (not 1 <= args.workers <= 32 or not 1 <= args.http_workers <= 32 or args.max_queue < 1 or
            not 1 <= args.read_concurrency <= 32 or
            args.read_pool_size is not None and not 0 <= args.read_pool_size <= args.read_concurrency or
            secret is not None and len(secret.encode()) < 32 or
            not 0 <= args.wal_size_limit <= 1 << 40 or
            not math.isfinite(args.stall_stale_seconds) or args.stall_stale_seconds <= 0):
        parser.error("执行体/读并发须为 1~32，连接池须为 0~读并发，密钥至少 32 字节，"
                     "WAL 上限须为 0~1TiB，停滞阈值须为正数")
    if args.role == "all" and args.http_workers != 1:
        parser.error("all 角色会启动本地准入执行体，禁止配置多个 HTTP 进程；请改用 api + amgl-worker")
    if args.http_workers > 1 and secret is None:
        parser.error("多个 HTTP 进程必须配置共享的 AMGL_CURSOR_SECRET")
    import uvicorn
    if args.http_workers == 1:
        app = create_app(args.db, tenant_id=args.tenant, api_token=token,
                         admin_token=admin_token, workers=args.workers, max_queue=args.max_queue,
                         read_concurrency=args.read_concurrency, read_pool_size=args.read_pool_size,
                         read_timings=args.read_timings,
                         cursor_secret=secret.encode() if secret is not None else None, role=args.role,
                         wal_size_limit_bytes=args.wal_size_limit,
                         stall_stale_seconds=args.stall_stale_seconds)
        uvicorn.run(app, host=args.host, port=args.port)
        return
    os.environ.update({
        "AMGL_RUNTIME_DB": args.db,
        "AMGL_RUNTIME_TENANT": args.tenant,
        "AMGL_RUNTIME_ROLE": args.role,
        "AMGL_RUNTIME_ADMISSION_WORKERS": str(args.workers),
        "AMGL_RUNTIME_MAX_QUEUE": str(args.max_queue),
        "AMGL_RUNTIME_READ_CONCURRENCY": str(args.read_concurrency),
        "AMGL_RUNTIME_READ_POOL_SIZE": "" if args.read_pool_size is None else str(args.read_pool_size),
        "AMGL_RUNTIME_READ_TIMINGS": "1" if args.read_timings else "0",
        "AMGL_RUNTIME_WAL_LIMIT": str(args.wal_size_limit),
        "AMGL_RUNTIME_STALL_STALE_SECONDS": str(args.stall_stale_seconds),
    })
    uvicorn.run("amgl.server:create_app_from_env", factory=True, workers=args.http_workers,
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
