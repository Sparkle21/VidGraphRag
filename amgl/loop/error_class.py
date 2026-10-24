"""异步工具的有限重试；只有调用方明确声明可安全重试才再次调用。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import random
import time


class ErrorClass(str, Enum):
    ENV = "env"
    SCHEMA = "schema"
    LOGIC = "logic"


class SchemaFailure(Exception):
    """由外部 schema/oracle 检查报告，不能靠模型自述生成。"""


class LogicFailure(Exception):
    """目标或业务校验失败，不就地重试。"""


def classify(error: Exception) -> ErrorClass:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return ErrorClass.ENV
    if isinstance(error, SchemaFailure):
        return ErrorClass.SCHEMA
    return ErrorClass.LOGIC


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 2
    total_timeout: float = 30

    def __post_init__(self):
        if self.max_attempts < 1 or self.base_delay < 0 or self.max_delay < self.base_delay or self.total_timeout <= 0:
            raise ValueError("重试参数超出范围")


def retry_policy(policy: RetryPolicy | None = None, *, safe_to_retry: bool = False,
                 classifier=classify, schema_feedback=None):
    """schema_feedback 异步回调必须从外部校验信号构造下一次调用的 kwargs。"""
    policy = policy or RetryPolicy()

    def decorate(fn):
        @wraps(fn)
        async def wrapped(*args, **kwargs):
            deadline = time.monotonic() + policy.total_timeout
            for attempt in range(1, policy.max_attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("工具调用总时限已耗尽")
                try:
                    return await asyncio.wait_for(fn(*args, **kwargs), remaining)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    category = classifier(error)
                    if not safe_to_retry or attempt == policy.max_attempts or category == ErrorClass.LOGIC:
                        raise
                    if category == ErrorClass.SCHEMA:
                        if schema_feedback is None:
                            raise
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("工具调用总时限已耗尽") from error
                        kwargs = await asyncio.wait_for(schema_feedback(error, dict(kwargs)), remaining)
                    elif category != ErrorClass.ENV:
                        raise
                    delay = random.uniform(0, min(policy.max_delay, policy.base_delay * 2 ** (attempt - 1)))
                    if deadline - time.monotonic() <= delay:
                        raise TimeoutError("工具调用总时限已耗尽") from error
                    await asyncio.sleep(delay)
        return wrapped
    return decorate
