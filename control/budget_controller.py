import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from __future__ import annotations
from pydantic import BaseModel


class BudgetExceededError(Exception):
    def __init__(self, reason: str, phase: str, current: float, limit: float):
        self.reason  = reason
        self.phase   = phase
        self.current = current
        self.limit   = limit
        super().__init__(
            f"[BudgetExceeded] phase={phase!r} reason={reason!r} "
            f"current={current} limit={limit}"
        )


class BudgetConfig(BaseModel):

    max_tokens_total:    int = 25_000
    max_tokens_planning: int = 8_000
    max_tokens_refining: int = 6_000  
    max_cost_usd:        float = 2.0    

    max_wall_seconds:    int = 1000   
    cost_per_1k_input_tokens:  float = 0.003   
    cost_per_1k_output_tokens: float = 0.015


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

@dataclass
class _JobBudget:
    config:BudgetConfig    
    tokens_used: Dict[str, TokenUsage]
    cost_usd:float = 0.0
    job_start_time:float = field(default_factory=time.monotonic)
    lock:asyncio.Lock = field(default_factory=asyncio.Lock)

    def total_tokens(self) -> int:
        return sum(self.tokens_used.values())

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.job_start_time
    

class BudgetController:
    def __init__(self, default_config: Optional[BudgetConfig] = None):
        self._default_config = default_config or BudgetConfig()
        self._jobs: Dict[str, _JobBudget] = {}


    def register_job(self,job_id: str,config_overrides: Optional[Dict] = None,) -> None:
        if config_overrides:
            cfg = self._default_config.model_copy(update=config_overrides)
        else:
            cfg=self._default_config

        self._jobs[job_id] = _JobBudget(config=cfg)

    def deregister_job(self,job_id:str)->None:
        self._jobs.pop(job_id,None)


    def assert_budget_available(self,job_id: str,phase: str,estimated_tokens: int = 0,) -> None:
        budget = self._get_budget(job_id)
        elapsed = budget.elapsed_seconds()
        if elapsed >= budget.config.max_wall_seconds:
            raise BudgetExceededError(
                reason="wall_clock_exceeded",
                phase=phase,
                current=elapsed,
                limit=budget.config.max_wall_seconds,
            )
        total = budget.total_tokens() + estimated_tokens
        if total > budget.config.max_tokens_total:
            raise BudgetExceededError(
                reason="total_tokens_exceeded",
                phase=phase,
                current=total,
                limit=budget.config.max_tokens_total,
            )
        
        phase_key = phase.upper()
        phase_limit = self._phase_limit(budget.config, phase_key)
        if phase_limit is not None:
            phase_used = budget.tokens_used.get(phase_key, 0) + estimated_tokens
            if phase_used > phase_limit:
                raise BudgetExceededError(
                    reason=f"{phase_key}_phase_tokens_exceeded",
                    phase=phase,
                    current=phase_used,
                    limit=phase_limit,
                )
        if budget.cost_usd >= budget.config.max_cost_usd:
            raise BudgetExceededError(
                reason="cost_exceeded",
                phase=phase,
                current=budget.cost_usd,
                limit=budget.config.max_cost_usd,
            )
        
    def record_usage(self,job_id: str,phase: str,input_tokens: int,output_tokens: int,) -> None:
        budget = self._get_budget(job_id)
        phase_key = phase.upper()
        total_tokens = input_tokens + output_tokens
        budget.tokens_used[phase_key] = budget.tokens_used.get(phase_key, 0) + total_tokens
        incremental_cost = (
            (input_tokens  / 1000) * budget.config.cost_per_1k_input_tokens +
            (output_tokens / 1000) * budget.config.cost_per_1k_output_tokens
        )
        budget.cost_usd += incremental_cost

    def get_usage_summary(self, job_id: str) -> Dict:
        budget = self._get_budget(job_id)
        return {
            "job_id":          job_id,
            "tokens_by_phase": dict(budget.tokens_used),
            "total_tokens":    budget.total_tokens(),
            "cost_usd":        round(budget.cost_usd, 6),
            "elapsed_seconds": round(budget.elapsed_seconds(), 2),
            "limits": {
                "max_tokens_total":  budget.config.max_tokens_total,
                "max_cost_usd":      budget.config.max_cost_usd,
                "max_wall_seconds":  budget.config.max_wall_seconds,
            },
        }

    def remaining_budget(self, job_id: str) -> Dict:
        budget = self._get_budget(job_id)
        return {
            "tokens_remaining": budget.config.max_tokens_total - budget.total_tokens(),
            "cost_remaining":   round(budget.config.max_cost_usd - budget.cost_usd, 6),
            "seconds_remaining": max(
                0,
                budget.config.max_wall_seconds - budget.elapsed_seconds()
            ),
        }


    def _get_budget(self, job_id: str) -> _JobBudget:
        budget = self._jobs.get(job_id)
        if budget is None:
            raise KeyError(
                f"BudgetController: job {job_id!r} not registered. "
                "Call register_job() before submitting."
            )
        return budget

    @staticmethod
    def _phase_limit(config: BudgetConfig, phase_key: str) -> Optional[int]:
        mapping = {
            "PLANNING": config.max_tokens_planning,
            "REFINING": config.max_tokens_refining,
        }
        return mapping.get(phase_key)