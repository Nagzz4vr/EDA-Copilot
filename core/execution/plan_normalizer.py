from typing import Dict, Any, List
import uuid


class PlanNormalizer:


    SAFE_ACTION_TYPES = {
        "impute_mean",
        "impute_median",
        "impute_mode",
        "impute_knn",
        "impute_constant",
        "one_hot_encode",
        "label_encode",
        "ordinal_encode",
        "target_encode",
        "scale_standard",
        "scale_minmax",
        "scale_robust",
        "drop_column",
        "drop",
        "cast_dtype",
        "log_transform",
    }

    NUMERIC_ONLY = {
        "impute_mean",
        "impute_median",
        "impute_knn",
        "scale_standard",
        "scale_minmax",
        "scale_robust",
        "log_transform",
    }

    CATEGORICAL_ONLY = {
        "one_hot_encode",
        "label_encode",
        "ordinal_encode",
        "target_encode",
    }

    def __init__(self, canonical_context: Dict[str, Any]):
        self.col_types = self._extract_column_types(canonical_context)


    def normalize(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        actions = plan.get("actions", [])
        normalized_actions = []

        for action in actions:
            action = self._ensure_structure(action)
            action = self._validate_action_type(action)

            if not action:
                continue

            action = self._fix_missing_columns(action)
            action = self._repair_type_conflicts(action)

            normalized_actions.append(action)

        plan["actions"] = normalized_actions
        plan["normalized"] = True
        return plan


    def _ensure_structure(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return None

        action.setdefault("id", self._generate_id(action))
        action.setdefault("action_type", "UNKNOWN")
        action.setdefault("target_columns", [])
        action.setdefault("params", {})
        action.setdefault("metadata", {})

        return action

    def _generate_id(self, action: Dict[str, Any]) -> str:
        base = action.get("action_type", "action")
        cols = "_".join(action.get("target_columns", ["col"]))
        return f"{base}_{cols}_{uuid.uuid4().hex[:6]}"


    def _validate_action_type(self, action: Dict[str, Any]) -> Dict[str, Any] | None:
        if action["action_type"] not in self.SAFE_ACTION_TYPES:
            return None
        return action


    def _fix_missing_columns(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not action["target_columns"]:
            action["target_columns"] = ["__default__"]
        return action

    def _repair_type_conflicts(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action["action_type"]
        cols = action.get("target_columns", [])

        for col in cols:
            col_type = self.col_types.get(col)

            # numeric-only protection
            if action_type in self.NUMERIC_ONLY and col_type == "categorical":
                action["metadata"]["auto_fixed"] = "demoted_to_safe_impute"
                action["action_type"] = "impute_mode"

            # categorical warning only (no mutation)
            if action_type in self.CATEGORICAL_ONLY and col_type == "numeric":
                action["metadata"]["warning"] = f"{col} is numeric but encoded"

        return action

    def _extract_column_types(self, ctx: Dict[str, Any]) -> Dict[str, str]:
        """
        Expected format:
        canonical_context["columns"] = [
            {"name": "age", "type": "numeric"},
            {"name": "gender", "type": "categorical"}
        ]
        """
        cols = ctx.get("columns", [])
        return {
            c.get("name"): c.get("type", "unknown")
            for c in cols if isinstance(c, dict)
        }