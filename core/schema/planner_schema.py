from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    action_type: str
    target_columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)

    metadata: Dict[str, Any] = Field(default_factory=dict)

class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    action_type: str
    target_columns: List[str]
    benefit_score: float
    cost_score: float
    risk_level: str
    originating_rule: str

    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    edge_type: str



class SignalGraph(BaseModel):

    model_config = ConfigDict(extra="forbid")
    nodes: list[dict]
    edges: list[Edge]

class OptimizedPlan(BaseModel):

    model_config = ConfigDict(extra="forbid")
    version:                 int            = 1
    actions:                 List[Action]
    metadata:                Dict[str, Any] = Field(default_factory=dict)
    speculative_candidates:  List[Any]      = Field(default_factory=list)

class ValidationReport(BaseModel):

    model_config = ConfigDict(extra="forbid")
    passed:     bool
    violations: List[Dict[str, Any]] = Field(default_factory=list)

class PlannerBundle(BaseModel):

    signal_graph:       SignalGraph
    optimized_plan:     OptimizedPlan
    validation_report:  Optional[ValidationReport] = None
    fingerprint:        str
    state_uuid:         str



class ExecutionInput(BaseModel):

    optimized_plan: OptimizedPlan
    signal_graph:   SignalGraph


class Signal(BaseModel):
    """Individual signal detected from dataset analysis"""
    signal_id: str
    signal_type: str  # e.g., "MISSING_VALUES", "HIGH_CARDINALITY", "CLASS_IMBALANCE"
    severity: str     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    confidence: float
    affected_columns: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SignalBag(BaseModel):
    """Collection of signals from the SIGNAL stage"""
    signals: List[Signal]
    dataset_health: Dict[str, float]
    timestamp: float = Field(default_factory=lambda: time.time())


class DecisionPlan(BaseModel):
    """Unordered/loosely-ordered LLM decisions from PLAN stage"""
    actions: List[Action]
    confidence_score: float
    reasoning: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentReview(BaseModel):
    """Human-in-the-loop or LLM audit result from REVIEW stage"""
    approved: bool
    overrides: List[Dict[str, Any]] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None
    amended_plan: Optional[Dict] = None
    reviewer_notes: str = ""


class ImpactReport(BaseModel):
    """Simulation results from SIMULATE stage"""
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    anomaly_flags: List[str] = Field(default_factory=list)
    risk_metrics: Dict[str, float] = Field(default_factory=dict)
    data_loss_estimate: float = 0.0
    simulation_timestamp: float = Field(default_factory=lambda: time.time())