"""可恢复的有限工具步骤运行器；模型/业务规划通过调用方提供的步骤接入。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Awaitable, Callable

from amgl.errors import Conflict, NotFound, UnsafeResume
from amgl.memory.admission.pipeline import AdmissionPipeline
from amgl.memory.audit import canonical
from amgl.memory.envelope import CandidateMemory
from .recovery import RecoveryStore, idempotency_key


@dataclass(frozen=True)
class Step:
    name: str
    operation: Callable[[dict, str], Awaitable[object]]


class LoopRuntime:
    def __init__(self, recovery: RecoveryStore, *, max_steps: int = 32, timeout: float = 60):
        if max_steps < 1 or timeout <= 0:
            raise ValueError("步骤和时间预算必须大于零")
        self.recovery, self.max_steps, self.timeout = recovery, max_steps, timeout

    async def run(self, run_id: str, steps: list[Step], inputs: dict, *, plan_version: str = "1") -> dict:
        if not steps or len(steps) > self.max_steps or len({s.name for s in steps}) != len(steps):
            raise ValueError("步骤必须非空、名称唯一且不超过预算")
        signature = hashlib.sha256(canonical([inputs, plan_version, [s.name for s in steps]]).encode()).hexdigest()
        try:
            state, version = await asyncio.to_thread(self.recovery.restore, run_id)
            if state["signature"] != signature:
                raise Conflict("恢复输入或步骤版本与检查点不一致")
        except NotFound:
            state = {"signature": signature, "next_step": 0, "outputs": [],
                     "status": "running", "deadline": time.time() + self.timeout}
            version = await asyncio.to_thread(self.recovery.snapshot, run_id, state)
        if state["status"] == "completed":
            return state
        while state["next_step"] < len(steps):
            remaining = state["deadline"] - time.time()
            if remaining <= 0:
                raise TimeoutError("运行总时限已耗尽")
            step = steps[state["next_step"]]
            context = json.loads(canonical({"inputs": inputs, "outputs": state["outputs"], "signature": signature}))

            async def invoke(key):
                return await step.operation(context, key)

            try:
                result = await asyncio.wait_for(self.recovery.execute(run_id, step.name, context, invoke), remaining)
            except Exception as error:
                state["status"] = "needs_reconciliation" if isinstance(error, UnsafeResume) else "interrupted"
                await asyncio.to_thread(self.recovery.snapshot, run_id, state, expected_version=version)
                raise
            state["outputs"].append(result)
            state["next_step"] += 1
            state["status"] = "completed" if state["next_step"] == len(steps) else "running"
            version = await asyncio.to_thread(self.recovery.snapshot, run_id, state, expected_version=version)
        return state

    async def remember(self, pipeline: AdmissionPipeline, run_id: str, slot: str, candidate: CandidateMemory):
        """回合结束的记忆提交可安全重复调用，使用与运行标识绑定的幂等键。"""
        if pipeline.tenant_id != self.recovery.tenant:
            raise ValueError("运行与记忆租户不一致")
        return await pipeline.submit(candidate, idempotency_key(self.recovery.tenant, run_id, "memory:" + slot))
