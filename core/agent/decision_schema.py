from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class ActionOverride(BaseModel):
    action_id: str = Field(..., description="The unique ID of the action node from the PlanOptimizer")
    new_action_type: str = Field(..., description="The replacement operation (e.g., 'ordinal_encode' instead of 'one_hot')")
    new_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Updated hyperparams for the tool")
    override_reason: str = Field(..., description="Why the automated suggestion was rejected")

class AgentReview(BaseModel):
    approved: bool = Field(..., description="False if the plan is fundamentally broken and needs refinement")
    overrides: List[ActionOverride] = Field(default_factory=list, description="Surgical changes to specific plan steps")
    global_flags: List[str] = Field(..., description="Categorical signals like 'target_leakage_detected' or 'imbalanced_class'")
    confidence: float = Field(..., description="0.0 to 1.0 rating of the final strategy's safety")
    reasoning: str = Field(..., description="Justification for the review results")