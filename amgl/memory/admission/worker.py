"""独立准入执行进程；HTTP API 仅负责持久化提交，worker 消费共享 SQLite 租约队列。"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
import logging
import math
import os
from pathlib import Path
import signal
import socket
import sys

from ..backends import FileKVBackend
from ..envelope import new_id
from .pipeline import AdmissionPipeline

logger = logging.getLogger("amgl")


def validate_heartbeat(interval: float, ttl: float) -> None:
    if not (math.isfinite(interval) and math.isfinite(ttl)) or interval <= 0 or ttl < 2 * interval:
        raise ValueError("心跳参数须为有限正数，有效期至少为间隔的 2 倍")


async def run_worker(db_path: str | Path, tenant_id: str = "local", *, workers: int = 2,
                     judge=None, judge_timeout: float = 5, lease_seconds: float = 30,
                     poll_interval: float = 0.1, heartbeat_interval: float = 2,
                     heartbeat_ttl: float = 10, checkpoint_interval: float = 30,
                     ready_event=None, shutdown_event=None,
                     on_started: Callable[[], None] | None = None) -> None:
    """运行到收到信号或外部事件；正常关停会等待已取得租约的任务完成。

    心跳过期即视为执行体不可用；本地准入协程异常结束时停止续约并以异常退出，交由进程管理器重启。
    心跳携带累计完成数与最近完成时间，供 API 区分“进程存活”与“任务在推进”；
    周期性执行 WAL 检查点，配合 journal_size_limit 控制磁盘占用。
    """
    validate_heartbeat(heartbeat_interval, heartbeat_ttl)
    if not (math.isfinite(checkpoint_interval) and checkpoint_interval >= 0):
        raise ValueError("WAL 检查点间隔必须为非负有限数，0 表示禁用")
    backend = FileKVBackend(db_path)
    pipeline = AdmissionPipeline(backend, tenant_id, judge=judge, workers=workers,
                                 judge_timeout=judge_timeout, lease_seconds=lease_seconds,
                                 poll_interval=poll_interval)
    worker_id, host, pid = new_id(), socket.gethostname(), os.getpid()

    def beat(status: str = "running") -> None:
        backend.heartbeat(tenant_id, worker_id, pid=pid, host=host, concurrency=workers,
                          ttl=heartbeat_ttl, status=status,
                          completed=pipeline.completed_count,
                          last_finished=pipeline.last_finished_at)

    local_stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals = []
    if shutdown_event is None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, local_stop.set)
                installed_signals.append(signum)
            except (NotImplementedError, RuntimeError):
                pass

    failed, registered = False, False
    try:
        await pipeline.start()
        # 首次注册失败直接退出：未登记的执行体不能对外声明就绪。
        await asyncio.to_thread(beat)
        registered = True
        if ready_event is not None:
            ready_event.set()
        if on_started is not None:
            on_started()
        next_beat = loop.time() + heartbeat_interval
        next_checkpoint = loop.time() + checkpoint_interval if checkpoint_interval else None
        while not local_stop.is_set() and (shutdown_event is None or not shutdown_event.is_set()):
            if not pipeline.healthy:
                logger.error("准入协程异常结束，停止心跳并退出")
                failed = True
                break
            if loop.time() >= next_beat:
                try:
                    await asyncio.to_thread(beat)
                except Exception:
                    # 单次续约失败由有效期吸收；持续失败会自然过期并被 API 判为未就绪。
                    logger.warning("准入执行体心跳写入失败，将在下个周期重试")
                next_beat = loop.time() + heartbeat_interval
            if next_checkpoint is not None and loop.time() >= next_checkpoint:
                try:
                    await asyncio.to_thread(backend.checkpoint_wal)
                except Exception:
                    # 检查点失败不影响任务处理；WAL 截断可由下次周期或写连接自动推进重试。
                    logger.warning("WAL 检查点执行失败，将在下个周期重试")
                next_checkpoint = loop.time() + checkpoint_interval
            try:
                await asyncio.wait_for(local_stop.wait(), min(poll_interval, 0.1, heartbeat_interval))
            except TimeoutError:
                pass
    finally:
        # 先关闭领新任务入口，持久化排空状态即使等待 SQLite 写锁也不会继续领任务。
        pipeline.request_stop()
        try:
            if registered:
                try:
                    await asyncio.to_thread(beat, "stopping")
                except Exception:
                    logger.warning("准入执行体无法登记排空状态")
            await pipeline.stop()
        finally:
            try:
                # 关闭异常也要注销；首次注册失败不创建新的排空记录。
                if registered:
                    await asyncio.to_thread(backend.deregister, tenant_id, worker_id)
            except Exception:
                logger.warning("准入执行体注销失败，记录将在心跳过期后失效")
            finally:
                for signum in installed_signals:
                    loop.remove_signal_handler(signum)
    if failed:
        raise RuntimeError("准入执行体异常退出")


def main() -> None:
    parser = argparse.ArgumentParser(description="AMGL 独立记忆准入执行体")
    parser.add_argument("--db", default="storage/amgl/amgl.sqlite3")
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--workers", default=2, type=int, help="本进程准入协程数量")
    parser.add_argument("--judge-timeout", default=5.0, type=float)
    parser.add_argument("--lease-seconds", default=30.0, type=float)
    parser.add_argument("--poll-interval", default=0.1, type=float)
    parser.add_argument("--heartbeat-interval", default=2.0, type=float, help="心跳续约间隔（秒）")
    parser.add_argument("--heartbeat-ttl", default=10.0, type=float, help="心跳有效期（秒），至少为间隔的 2 倍")
    parser.add_argument("--checkpoint-interval", default=30.0, type=float,
                        help="WAL 检查点间隔（秒），0 为禁用")
    args = parser.parse_args()
    try:
        validate_heartbeat(args.heartbeat_interval, args.heartbeat_ttl)
    except ValueError as error:
        parser.error(str(error))
    if not (math.isfinite(args.checkpoint_interval) and args.checkpoint_interval >= 0):
        parser.error("--checkpoint-interval 必须为非负有限数")
    try:
        asyncio.run(run_worker(args.db, args.tenant, workers=args.workers,
                               judge_timeout=args.judge_timeout, lease_seconds=args.lease_seconds,
                               poll_interval=args.poll_interval, heartbeat_interval=args.heartbeat_interval,
                               heartbeat_ttl=args.heartbeat_ttl, checkpoint_interval=args.checkpoint_interval))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except RuntimeError as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
