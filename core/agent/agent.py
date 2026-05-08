import json
from litellm import completion
from .decision_schema import AgentReview
from observability.trace_logger import TraceLogger

class ReviewerAgent:
    def __init__(self, model: str = "gpt-4o", session_id: str = "unset"):
        self.model = model
        self.session_id = session_id
        self.logger = TraceLogger(session_id=session_id)

    def _build_system_prompt(self) -> str:
        return """
            You are a Data Plan Reviewer (not a planner).

            INPUT:
            - GlobalImpactReport: expected dataset-level effects
            - ProposedPlan: ordered actions with action_id, type, parameters

            TASK:
            Audit the plan for correctness and risk.

            CHECK:
            1. Logical consistency (conflicting or redundant actions)
            2. Data integrity risks (excessive data loss, distortion)
            3. Statistical validity (wrong transformations/imputations)
            4. Leakage or target misuse
            5. Alignment with GlobalImpactReport

            ACTIONS:
            - Approve if plan is sound
            - Override only if a specific action is incorrect
            - Reject if plan is unsafe

            CONSTRAINTS:
            - Do NOT create new actions
            - Do NOT redesign the plan
            - Overrides must reference existing action_id
            - Keep overrides minimal and precise

            OUTPUT:
            Return valid JSON (AgentReview schema) with:
            - approved (bool)
            - overrides (list of targeted fixes)
            - global_flags (predefined risk signals if any)
            - confidence (0–1)
            - reasoning (concise, reference action_id where relevant)
                """.strip()
    
    def review_plan(self,impact_report:dict,optimized_plan:dict)->AgentReview:
        prompt={
            "simulation_results":impact_report,
            "proposed_action":optimized_plan
        }
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": json.dumps(prompt)}
            ],
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        review_dict = json.loads(raw_content)
        review = AgentReview(**review_dict)

        self.logger.log(
            tool="reviewer_agent",
            intent="plan_audit",
            inputs={"plan_version": optimized_plan.get("version")},
            outputs=review.dict(),
            confidence=review.confidence
        )

        return review