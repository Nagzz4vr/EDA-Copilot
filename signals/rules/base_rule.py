from abc import ABC, abstractmethod
from typing import Dict, Any,Literal
from dataclasses import dataclass, asdict
from datetime import datetime



@dataclass
class RuleOutput:
    """Canonical signal output from any rule"""
    rule_name: str
    priority: int
    type: Literal["feature_reduction", "feature_review", "data_quality", "model_risk"]
    action: str
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    details: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
class BaseRule(ABC):
    name: str = "base_rule"
    priority: int = 0  # higher runs first

    @abstractmethod
    def applies(self, context: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> RuleOutput:
        pass
