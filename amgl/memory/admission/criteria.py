"""四判据评估扩展点。未配置评估器时明确降级，绝不伪造 LLM 分数。"""
from typing import Protocol
from pydantic import Field

from ..envelope import CandidateMemory, Model


class CriteriaScores(Model):
    stability: float = Field(ge=0, le=1, allow_inf_nan=False)
    reusability: float = Field(ge=0, le=1, allow_inf_nan=False)
    review_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    abstraction: float = Field(ge=0, le=1, allow_inf_nan=False)

    @property
    def score(self) -> float:
        return sum(self.model_dump().values()) / 4


class CriteriaJudge(Protocol):
    async def evaluate(self, candidate: CandidateMemory) -> CriteriaScores: ...
