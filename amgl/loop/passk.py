"""任务成功率 pass^k 评测，不把答案相同当作成功，也不混淆 pass@k。"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass
from math import comb
from typing import Callable


@dataclass
class PassKResult:
    k: int
    trials_per_task: int
    pass_1: float
    pass_k: float
    outcomes: list[list[bool]]
    errors: list[list[str | None]]

    def to_dict(self) -> dict:
        return asdict(self)


async def run_passk(tasks: list, task_fn: Callable, judge_fn: Callable, *, k: int = 3,
                   trials: int | None = None, concurrency: int = 1, timeout: float = 30,
                   reset_fn: Callable | None = None) -> PassKResult:
    """task_fn/judge_fn/reset_fn 均为异步函数。

    每次深拷贝任务输入；有数据库等外部状态时，reset_fn 必须返回隔离的任务环境。
    trials=n >= k，以 C(s,k)/C(n,k) 估计同任务 k 次全部成功；n=k 时为全成功指示值。
    """
    n = trials if trials is not None else k
    if not tasks or not 1 <= k <= n or concurrency < 1 or timeout <= 0:
        raise ValueError("评测必须有任务，且满足 1 <= k <= trials")
    semaphore = asyncio.Semaphore(concurrency)

    async def attempt(task):
        async with semaphore:
            phase = "reset"
            async def evaluate():
                nonlocal phase
                isolated = copy.deepcopy(task)
                if reset_fn is not None:
                    isolated = await reset_fn(isolated)
                phase = "task"
                answer = await task_fn(isolated)
                phase = "judge"
                verdict = await judge_fn(isolated, answer)
                if not isinstance(verdict, bool):
                    raise TypeError("评测器必须返回布尔值")
                return verdict
            try:
                return await asyncio.wait_for(evaluate(), timeout), None
            except asyncio.CancelledError:
                raise
            except Exception as error:
                return False, phase + ":" + type(error).__name__

    rows = []
    for task in tasks:
        rows.append(await asyncio.gather(*(attempt(task) for _ in range(n))))
    outcomes = [[passed for passed, _ in row] for row in rows]
    errors = [[error for _, error in row] for row in rows]
    successes = [sum(row) for row in outcomes]
    return PassKResult(k=k, trials_per_task=n,
                       pass_1=sum(successes) / (n * len(tasks)),
                       pass_k=sum(comb(s, k) / comb(n, k) if s >= k else 0 for s in successes) / len(tasks),
                       outcomes=outcomes, errors=errors)
