
import os
import json
import asyncio
import time
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from litellm import completion
from litellm.exceptions import RateLimitError, NotFoundError, ServiceUnavailableError
from dotenv import load_dotenv
from .decision_schema import AgentReview
from observability.trace_logger import TraceLogger

load_dotenv()


MODEL_POOL: List[str] = [
    "groq/llama-3.3-70b-versatile",  # High quality for planning
    "groq/llama-3.1-8b-instant",    # Fast fallback
    "groq/qwen-2.5-coder-32b",       # Good for logic/JSON
]
RETRY_AFTER_SECONDS = 10   



class PlanResult(BaseModel):
    actions:          list[dict]
    confidence_score: float = Field(ge=0.0, le=1.0)
    tokens_used:      int   = 0
    reasoning:        str   = ""
    metadata:         dict  = Field(default_factory=dict)
    model_used:       str   = ""   # for observability



class PlannerAgent:
    def __init__(self, session_id: str = "unset"):
        self.api_key    = os.environ["GROQ_API_KEY"]
        self.session_id = session_id
        self.headers    = {
            "HTTP-Referer": os.getenv("SITE_URL", "http://localhost"),
            "X-Title":      os.getenv("SITE_NAME", "ml-pipeline"),
        }
        self.logger        = TraceLogger(session_id=session_id)
        self._model_pool   = MODEL_POOL.copy()
        self._model_index  = 0   # tracks current model across calls

    @property
    def _current_model(self) -> str:
        return self._model_pool[self._model_index]

    def _rotate_model(self) -> str:
        """Advance to next model in pool. Returns the new model string."""
        self._model_index = (self._model_index + 1) % len(self._model_pool)
        new_model = self._current_model
        self.logger.log(
            tool="MODEL_ROTATOR",
            intent=f"Rotated to {new_model}",
            inputs={}, outputs={"model": new_model}, confidence=1.0,
        )
        return new_model

    async def _call_with_fallback(self, messages: list, context: str) -> tuple[Any, str]:
        attempts = 0
        last_error = None

        while attempts < len(self._model_pool):
            model = self._current_model
            try:
                response = await asyncio.to_thread(
                    completion,
                    model=model,
                    api_key=self.api_key,
                    messages=messages,
                    response_format={"type": "json_object"},
                    extra_headers=self.headers,
                )
                return response, model

            except (RateLimitError, NotFoundError, ServiceUnavailableError) as e:
                last_error = e
                self.logger.log(
                    tool="MODEL_ROTATOR",
                    intent=f"{type(e).__name__} on {model} ({context}) — rotating",
                    inputs={}, outputs={"error": str(e)}, confidence=0.5,
                )
                self._rotate_model()
                await asyncio.sleep(RETRY_AFTER_SECONDS)
                attempts += 1

            except Exception as e:
                raise

        raise RuntimeError(
            f"All {len(self._model_pool)} models exhausted for {context}. "
            f"Last error: {last_error}"
        )



    def _build_planner_system_prompt(self) -> str:
        return """
You are a Data Transformation Planner for ML pipelines.

INPUT you will receive:
- signal_graph:       nodes describing data quality issues detected
- canonical_context:  dataset statistics, column types, target column
- rejection_feedback: (optional) why the previous plan was rejected

YOUR TASK:
Produce a transformation plan that resolves the signals.

RULES:
- Only use these action_types:
    impute_mean, impute_median, impute_mode, impute_constant,
    drop_column, drop_rows, one_hot_encode, ordinal_encode,
    label_encode, scale_standard, scale_minmax, scale_robust,
    log_transform, clip_outliers, remove_outliers,
    type_cast, rename_column, fill_forward, fill_backward
- Every action must have a unique action_id (e.g. "act_001")
- Never touch the target column unless imputing it
- If rejection_feedback is present, directly address the concerns raised

OUTPUT FORMAT — return only valid JSON, no markdown, no explanation:
{
  "actions": [
    {
      "action_id":   "act_001",
      "action_type": "impute_median",
      "column":      "age",
      "parameters":  {},
      "rationale":   "35% missing values, numeric column"
    }
  ],
  "confidence_score": 0.85,
  "reasoning": "Brief explanation of the overall strategy"
}
        """.strip()

    async def generate_plan(
        self,
        signal_graph:       Any,
        canonical_context:  Any,
        rejection_feedback: Optional[dict] = None,
    ) -> PlanResult:

        graph_dict = (
            signal_graph.model_dump()
            if hasattr(signal_graph, "model_dump")
            else signal_graph
        )
        context_dict = (
            canonical_context.canonical_data.model_dump()
            if hasattr(canonical_context, "canonical_data")
            else canonical_context
        )

        payload: dict = {
            "signal_graph":      graph_dict,
            "canonical_context": context_dict,
        }
        if rejection_feedback:
            payload["rejection_feedback"] = rejection_feedback

        messages = [
            {"role": "system", "content": self._build_planner_system_prompt()},
            {"role": "user",   "content": json.dumps(payload)},
        ]

        response, model_used = await self._call_with_fallback(messages, context="planner")

        raw_content = response.choices[0].message.content
        tokens_used = getattr(response.usage, "total_tokens", 0)

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"PlannerAgent ({model_used}) returned non-JSON: {e}\n"
                f"Raw: {raw_content[:500]}"
            )

        if "actions" not in parsed:
            raise RuntimeError(
                f"PlannerAgent response missing 'actions'. "
                f"Got keys: {list(parsed.keys())}"
            )

        self.logger.log(
            tool="planner_agent",
            intent="plan_generated",
            inputs={"has_rejection_feedback": rejection_feedback is not None},
            outputs={
                "action_count":     len(parsed["actions"]),
                "confidence_score": parsed.get("confidence_score", 0.0),
                "model_used":       model_used,
            },
            confidence=parsed.get("confidence_score", 0.0),
        )
        raw_actions = parsed["actions"]
        print("RAW ACTIONS VALUE:", repr(parsed["actions"]))

        # Model sometimes returns actions as a JSON string instead of a list
        if isinstance(raw_actions, str):
            try:
                raw_actions = json.loads(raw_actions)
            except json.JSONDecodeError:
                # Try stripping leading colon if model prefixed it e.g. ":[{..."
                stripped = raw_actions.lstrip(": \n")
                raw_actions = json.loads(stripped)

        if not isinstance(raw_actions, list):
            raise RuntimeError(
                f"PlannerAgent 'actions' is {type(raw_actions).__name__} after parsing — "
                f"expected list. Value: {str(raw_actions)[:200]}"
            )

        return PlanResult(
            actions=         raw_actions,
            confidence_score= float(parsed.get("confidence_score", 0.7)),
            tokens_used=      tokens_used,
            reasoning=        parsed.get("reasoning", ""),
            metadata=         parsed.get("metadata", {}),
            model_used=       model_used,
        )



    def _build_reviewer_system_prompt(self) -> str:
        return """
You are a Data Plan Reviewer (not a planner).

INPUT:
- simulation_results: expected dataset-level effects (may be null on first review)
- proposed_action:    ordered actions with action_id, type, parameters
- rejection_feedback: (optional) previous rejection reasons if this is a replan

TASK:
Audit the plan for correctness and risk.

CHECK:
1. Logical consistency (conflicting or redundant actions)
2. Data integrity risks (excessive data loss, distortion)
3. Statistical validity (wrong transformations/imputations)
4. Leakage or target misuse
5. Alignment with simulation_results if available
6. If replanning: verify previously flagged issues are addressed

ACTIONS:
- Approve if plan is sound
- Override only if a specific action is incorrect (reference its action_id)
- Reject if plan is fundamentally unsafe

CONSTRAINTS:
- Do NOT create new actions
- Overrides must reference existing action_id

OUTPUT — return only valid JSON, no markdown:
{
  "approved": true,
  "overrides": [],
  "global_flags": [],
  "confidence": 0.9,
  "reasoning": "Plan is sound."
}
        """.strip()

    async def review_plan(
        self,
        impact_report:      Any,
        optimized_plan:     dict,
        rejection_feedback: Optional[Dict[str, Any]] = None,
    ) -> AgentReview:

        prompt: dict = {
            "simulation_results": impact_report or "Not yet available — pre-simulation review",
            "proposed_action":    optimized_plan,
        }
        if rejection_feedback:
            prompt["rejection_feedback"] = rejection_feedback
            prompt["is_replan"]          = True

        messages = [
            {"role": "system", "content": self._build_reviewer_system_prompt()},
            {"role": "user",   "content": json.dumps(prompt, indent=2)},
        ]

        response, model_used = await self._call_with_fallback(messages, context="reviewer")

        raw_content = response.choices[0].message.content

        try:
            review_dict = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"ReviewerAgent ({model_used}) returned non-JSON: {e}\n"
                f"Raw: {raw_content[:500]}"
            )

        review = AgentReview.model_validate(review_dict)

        self.logger.log(
            tool="reviewer_agent",
            intent="plan_audit",
            inputs={
                "plan_version": optimized_plan.get("version"),
                "is_replan":    rejection_feedback is not None,
                "model_used":   model_used,
            },
            outputs=review.model_dump(),
            confidence=review.confidence,
        )

        return review