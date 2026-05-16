import pandas as pd
from typing import Dict, Any
from copy import deepcopy

import asyncio
from typing import Any, Dict, Optional
 
from app.config import HITL_POLL_INTERVAL, HITL_TIMEOUT_SECONDS
from core.hitl import HitlStore


class HITLEscalationError(Exception):
    pass

class HITLRequiredError(Exception):
    """Raised when human approval is required."""
    pass


class RiskEngine:
    """
    Determines whether a transformation plan
    requires human approval.
    """

    HIGH_RISK_OPERATIONS = {
        "drop_column",
        "drop_rows",
        "remove_outliers",
        "target_encode",
        "schema_mutation",
        "delete_feature",
        "impute_median",   # temporary for testing
    }

    def __init__(self, logger):
        self.logger = logger

    def assess(
        self,
        plan: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            "risk_level": "LOW" | "MEDIUM" | "HIGH",
            "requires_hitl": bool,
            "reasons": list[str]
        }
        """

        metadata = metadata or {}

        transforms = plan.get("transforms") or plan.get("actions") or []

        reasons = []
        risk_level = "LOW"

        for transform in transforms:
            operation = transform.get("operation")
            if operation in self.HIGH_RISK_OPERATIONS:
                risk_level = "HIGH"
                reasons.append(f"High-risk operation detected: {operation}")
            if (operation == "drop_column"and transform.get("column")== metadata.get("target_column")):
                risk_level = "HIGH"
                reasons.append("Attempting to drop target column")
            if (operation == "impute"and transform.get("missing_ratio", 0) > 0.6):
                if risk_level != "HIGH":
                    risk_level = "MEDIUM"
                reasons.append("Large missing-value imputation")
        result = {"risk_level": risk_level,"requires_hitl": risk_level == "HIGH","reasons": reasons}
        self.logger.log(
            tool="RISK_ENGINE",
            intent="Risk assessment completed",
            inputs={
                "num_transforms": len(transforms)
            },
            outputs=result,
            confidence=0.9
        )
        return result

class HITLGate:
    def __init__(
        self,
        hitl_store: HitlStore,
        poll_interval: float = HITL_POLL_INTERVAL,
        timeout: int = HITL_TIMEOUT_SECONDS,
    ) -> None:
        self.store         = hitl_store
        self.poll_interval = poll_interval
        self.timeout       = timeout
 
    async def request_approval(
        self,
        plan: Dict[str, Any],
        risk_result: Dict[str, Any],
        state_uuid: str,
        diff_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pause the pipeline and wait for a human decision.
 
        Returns:
            {"action": "APPROVE"}
            {"action": "REJECT", "reason": "<optional text>"}
 
        Raises:
            asyncio.TimeoutError  — if no decision arrives within self.timeout
            RuntimeError          — if the store row disappears unexpectedly
        """
        job_id = state_uuid  # 1-to-1 mapping; state_uuid IS the job key
 
        # ── Write the request so Streamlit can find it ─────────────────────
        await asyncio.to_thread(
            self.store.write_request,
            job_id=job_id,
            state_uuid=state_uuid,
            plan_dict=plan,
            risk_result=risk_result,
            diff_data=diff_data,
        )
 
        # ── Poll until resolved or timed out ───────────────────────────────
        elapsed = 0.0
        while elapsed < self.timeout:
            await asyncio.sleep(self.poll_interval)
            elapsed += self.poll_interval
 
            row = await asyncio.to_thread(self.store.read_request, job_id)
 
            if row is None:
                raise RuntimeError(
                    f"HITLGate: store row for job {job_id!r} disappeared mid-poll. "
                    "Check HitlStore for concurrent deletion."
                )
 
            status = row.get("status", "PENDING")
 
            if status == "APPROVED":
                await asyncio.to_thread(self.store.clear, job_id)
                return {"action": "APPROVE"}
 
            if status == "REJECTED":
                reason = row.get("decision_reason")
                await asyncio.to_thread(self.store.clear, job_id)
                return {"action": "REJECT", "reason": reason}
 
            # Still PENDING — keep waiting

        # ── Timeout path ───────────────────────────────────────────────────
        await asyncio.to_thread(self.store.clear, job_id)
        raise asyncio.TimeoutError(
            f"HITLGate: no human decision received within {self.timeout}s "
            f"for job {job_id!r}"
        )
    
class HITL:
    """
    Human approval system for risky operations.
    """

    def __init__(self, logger):
        self.logger = logger

    async def request_approval(self,plan: Dict[str, Any],risk_result: Dict[str, Any],state_uuid: str) -> Dict[str, Any]:
        self.logger.log(
            tool="HITL",
            intent="Human approval requested",
            inputs={
                "state_uuid": state_uuid,
                "risk_level": risk_result["risk_level"]
            },
            outputs={},
            confidence=0.5
        )
        self._display_review(
            plan,
            risk_result
        )
        action = await self._await_user_input()
        self.logger.log(
            tool="HITL",
            intent=f"Human decision: {action}",
            inputs={"state_uuid": state_uuid},
            outputs={"action": action},
            confidence=1.0
        )
        return {
            "action": action,
            "state_uuid": state_uuid,
            "risk_level": risk_result["risk_level"]
                }

    def _display_review(
        self,
        plan: Dict[str, Any],
        risk_result: Dict[str, Any]
    ):
        """
        Display risky actions to the user.
        """

        print("\n" + "=" * 80)
        print("HUMAN APPROVAL REQUIRED")
        print("=" * 80)

        print(f"\nRisk Level: {risk_result['risk_level']}")

        print("\nReasons:")

        self.logger.log(
    tool="HITL_UI",
    intent="Displaying human review payload",
    inputs={
        "plan": plan,
        "risk_result": risk_result
    },
    outputs={},
    confidence=0.8
)
        for reason in risk_result["reasons"]:
            print(f" - {reason}")

        print("\nProposed Transformations:\n")

        for idx, transform in enumerate(plan.get("transforms", []),start=1):
            operation = transform.get( "operation", "UNKNOWN")
            column = transform.get("column","UNKNOWN")
            print(f"{idx}. {operation.upper()} -> {column}")
            rationale = transform.get("rationale")
            if rationale:
                print(f" Reason: {rationale}")

        print("\n[A] APPROVE")
        print("[R] REJECT")
        print("=" * 80)

    async def _await_user_input(self) -> str:
        """
        Wait for human response.
        """
        import asyncio
        while True:
            choice = await asyncio.to_thread(
                input,
                "\nYour choice (A/R): "
            )
            choice = choice.strip().upper()
            if choice == "A":
                return "APPROVE"
            elif choice == "R":
                return "REJECT"
            print("Invalid input.")

