
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class PlanMetadata:
    total_benefit: float
    total_cost: float
    max_risk: str
    estimated_memory_delta_mb: float

@dataclass(frozen=True)
class ExecutionPlan:
    version: int
    actions: List[Dict[str, Any]]
    metadata: PlanMetadata
    speculative_candidates: List[Dict[str, Any]]


