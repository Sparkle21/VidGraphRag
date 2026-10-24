"""有界异步执行体；回执只在事务持久化成功后返回。

超量拒绝并返回 429，不以无限内存队列或跳过安全闸宣称永不阻塞。
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

from amgl.errors import Conflict
from ..backends import FileKVBackend
from ..envelope import CandidateMemory, MemoryEnvelope, SourceTier, State, utcnow
from .criteria import CriteriaJudge, CriteriaScores
from .rules import classify_sensitivity, rejection_reason

logger = logging.getLogger("amgl")


class AdmissionPipeline:
    def __init__(self, backend: FileKVBackend, tenant_id: str = "local", *,
                 judge: CriteriaJudge | None = None, workers: int = 2,
                 max_queue: int = 10000, judge_timeout: float = 5,
                 lease_seconds: float = 30, poll_interval: float = 0.1):
        if not tenant_id or not 1 <= len(tenant_id) <= 128:
            raise ValueError("租户标识长度必须为 1~128")
        if (not all(math.isfinite(value) for value in (judge_timeout, lease_seconds, poll_interval)) or
                not 1 <= workers <= 32 or max_queue < 1 or judge_timeout <= 0 or poll_interval <= 0):
            raise ValueError("执行体配置超出范围")
        if lease_seconds <= judge_timeout + 5:
            raise ValueError("任务租约必须长于评估超时与存储等待窗口")
        self.backend, self.tenant_id, self.judge = backend, tenant_id, judge
        self.workers, self.max_queue = workers, max_queue
        self.judge_timeout, self.lease_seconds, self.poll_interval = judge_timeout, lease_seconds, poll_interval
        self._tasks: list[asyncio.Task] = []
        self._stop_event: asyncio.Event | None = None
        # 进度统计仅用于停滞告警：发生持久化状态转换（完成或释放）才计入。
        self.completed_count = 0
        self.last_finished_at: float | None = None

    async def submit(self, candidate: CandidateMemory, idempotency_key: str):
        if not 1 <= len(idempotency_key) <= 128:
            raise ValueError("幂等键长度必须为 1~128")
        # 先扫描，再将允许持久化的内容放进队列；拒绝回执只保留安全的原因码。
        rejected = rejection_reason(candidate)
        candidate = classify_sensitivity(candidate)
        return await asyncio.to_thread(self.backend.submit, self.tenant_id, candidate,
                                       idempotency_key, rejected=rejected, max_queue=self.max_queue)

    async def start(self):
        if self._tasks:
            raise RuntimeError("准入执行体已经启动")
        self._stop_event = asyncio.Event()
        self._tasks = [asyncio.create_task(self._worker(), name=f"amgl-admission-{i}") for i in range(self.workers)]

    def request_stop(self) -> None:
        """同步停止领新任务；允许调用方在等待持久化排空状态前先关闭准入入口。"""
        if self._stop_event is not None:
            self._stop_event.set()

    async def stop(self):
        if not self._tasks:
            return
        self.request_stop()
        tasks = self._tasks
        try:
            # 先让已取得租约的任务完成，避免正常关停退化为等待租约超时恢复。
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=self.judge_timeout + 6)
        except TimeoutError:
            pass
        finally:
            # 某一协程已取消或抛错也必须等其他协程退出，不能丢失仍在运行的任务引用。
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks = []
            self._stop_event = None

    @property
    def healthy(self) -> bool:
        return (self._stop_event is not None and not self._stop_event.is_set() and
                len(self._tasks) == self.workers and all(not task.done() for task in self._tasks))

    def progress(self) -> dict:
        """本地执行体完成进度；与 FileKVBackend.admission_progress 结构一致。"""
        now = time.time()
        return {"ready_workers": self.workers if self.healthy else 0,
                "completed_total": self.completed_count,
                "last_finished_seconds_ago": None if self.last_finished_at is None
                else round(max(0.0, now - self.last_finished_at), 3)}

    async def process_one(self) -> bool:
        job = await asyncio.to_thread(self.backend.claim, self.tenant_id, self.lease_seconds)
        if job is None:
            return False
        try:
            if job["attempts"] > 3:
                await asyncio.to_thread(self.backend.release, self.tenant_id, job)
                self._record_progress()
                return True
            envelope = MemoryEnvelope(**job["candidate"].model_dump(), tenant_id=self.tenant_id)
            envelope.lifecycle.state = State.PROVISIONAL
            envelope.lifecycle.state_entered_at = utcnow()
            envelope.governance.decided_by = "rules"
            envelope.governance.reason = "judge_unconfigured"
            if self.judge is not None:
                try:
                    scores = await asyncio.wait_for(self.judge.evaluate(job["candidate"]), self.judge_timeout)
                    scores = CriteriaScores.model_validate(scores)
                    envelope.governance.criteria_scores = scores.model_dump()
                    # 准入价值分不是事实真实性；评估器不能自行晋升状态。
                    envelope.trust.score = min(scores.score, 0.6)
                    envelope.governance.decided_by = "judge"
                    envelope.governance.reason = "criteria_scored"
                except Exception:
                    envelope.governance.reason = "judge_unavailable"
            if envelope.provenance.source_tier == SourceTier.THIRD_PARTY:
                envelope.trust.score = min(envelope.trust.score, 0.3)
            await asyncio.to_thread(self.backend.finish, self.tenant_id, job, envelope)
            self._record_progress()
        except Conflict:
            # 租约过期不能抢回所有权，等待新执行体处理。
            pass
        except Exception:
            logger.warning("准入任务处理失败，将按有限次数重试")
            await asyncio.to_thread(self.backend.release, self.tenant_id, job)
            self._record_progress()
        return True

    def _record_progress(self) -> None:
        self.completed_count += 1
        self.last_finished_at = time.time()

    async def _worker(self):
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                worked = await self.process_one()
                if not worked:
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), self.poll_interval)
                    except TimeoutError:
                        pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("准入存储暂不可用，执行体退避等待")
                try:
                    await asyncio.wait_for(self._stop_event.wait(), max(self.poll_interval, 0.5))
                except TimeoutError:
                    pass
