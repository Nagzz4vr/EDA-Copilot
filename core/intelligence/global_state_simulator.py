import pandas as pd
from typing import Dict, Any, List

from core.validation.leakage import LeakageDetector

LEAKY_GLOBAL_OPS = {
    "IMPUTE_MEAN",
    "IMPUTE_MEDIAN",
    "SCALE_STANDARD",
    "SCALE_MINMAX",
}

LEAKAGE_THRESHOLD_CRITICAL = 0.95
LEAKAGE_THRESHOLD_WARNING = 0.85


class GlobalStateSimulator:
    def __init__(
        self,
        plan: Dict[str, Any],
        dataset_sample: pd.DataFrame,
        dataset_schema: Dict[str, Any],
        max_col: int = 50,
    ):
        self.plan = plan
        self.dataset_sample = dataset_sample.copy()
        self.dataset_schema = dataset_schema
        self.simulated_df = dataset_sample.copy()
        self.max_col = max_col

        self.report = {
            "plan_invalid": False,
            "violations": [],
            "projections": {},
            "leakage_risks": [],
            "action_risks": [],
            "overall_risk_level": "LOW",
        }


    def _add_risk(self, bucket: str, entry: dict):
        if bucket not in self.report:
            self.report[bucket] = []
        self.report[bucket].append(entry)

    def simulate(self) -> Dict[str, Any]:
        try:
            self._run_metadata_projection()
            self._run_empirical_sampling()

            if self.report["plan_invalid"]:
                self.report["overall_risk_level"] = "HIGH"
                return self.report

            self._run_leakage_and_risk_analysis()
            self._rollup_risk_level()

        except Exception as e:
            self.report["plan_invalid"] = True
            self._add_risk("violations", {
                "type": "SIMULATOR_FATAL",
                "message": str(e),
            })
            self.report["overall_risk_level"] = "HIGH"

        return self.report


    def _safe_cols(self, action: Dict[str, Any]) -> List[str]:
        cols = action.get("target_columns", [])
        if cols is None:
            return []
        if isinstance(cols, str):
            return [cols]
        return [str(c) for c in cols]

    def _run_metadata_projection(self):
        current_count = self.dataset_schema["dataset_overview"]["num_columns"]

        for action in self.plan.get("actions", []):
            act_type = str(action.get("action_type", "")).upper()
            targets = self._safe_cols(action)

            if act_type in {"DROP", "DROP_COLUMN"}:
                current_count -= len(targets)

        current_count = max(0, current_count)

        self.report["projections"]["final_column_count"] = current_count

        num_rows = self.dataset_schema["dataset_overview"]["num_rows"]
        self.report["projections"]["memory_footprint_mb"] = (
            num_rows * current_count * 8 / (1024 * 1024)
        )


    def _run_empirical_sampling(self):
        last_action = None

        try:
            for action in self.plan.get("actions", []):
                last_action = action
                self.simulated_df = self.apply_single_action(
                    self.simulated_df, action
                )

            mem_mb = self.simulated_df.memory_usage(deep=True).sum() / (1024 * 1024)
            self.report["projections"]["actual_sample_memory_mb"] = mem_mb

        except Exception as e:
            self.report["plan_invalid"] = True
            self._add_risk("violations", {
                "type": "EMPIRICAL_EXECUTION_FAILED",
                "message": str(e),
                "offending_action": last_action,
            })


    def _run_leakage_and_risk_analysis(self):
        label_col = self.dataset_schema.get("label_column")

        if not label_col:
            self._add_risk("action_risks", {
                "type": "SKIPPED_LEAKAGE_CHECK",
                "reason": "No label_column in schema",
            })
            return

        self._run_target_leakage_check(label_col)
        self._run_temporal_leakage_check()
        self._run_action_ordering_check()

    def _run_target_leakage_check(self, label_col: str):
        if label_col not in self.simulated_df.columns:
            return

        for action in self.plan.get("actions", []):
            act_type = str(action.get("action_type", "")).upper()
            if act_type not in {"ENCODE", "IMPUTE", "SCALE", "CREATE_FEATURE"}:
                continue

            for col in self._safe_cols(action):
                if col not in self.simulated_df.columns:
                    continue

                score = LeakageDetector.calculate_appropriate_metric(
                    self.simulated_df[col],
                    self.simulated_df[label_col],
                )

                if abs(score) > LEAKAGE_THRESHOLD_CRITICAL:
                    level = "CRITICAL"
                elif abs(score) > LEAKAGE_THRESHOLD_WARNING:
                    level = "WARNING"
                else:
                    continue

                self._add_risk("leakage_risks", {
                    "column": col,
                    "level": level,
                    "score": round(score, 4),
                })

    def _run_temporal_leakage_check(self):
        for action in self.plan.get("actions", []):
            act_type = action.get("action_type")

            if act_type not in LEAKY_GLOBAL_OPS:
                continue

            if action.get("fit_on") != "train_only":
                for col in self._safe_cols(action):
                    self._add_risk("action_risks", {
                        "type": "TRAIN_TEST_LEAKAGE",
                        "column": col,
                        "action": act_type,
                    })


    def _run_action_ordering_check(self):
        risky_pairs = [
            ("IMPUTE", "DROP"),
            ("ENCODE", "SCALE"),
        ]

        col_seq = {}

        for action in self.plan.get("actions", []):
            act_type = action.get("action_type")
            for col in self._safe_cols(action):
                col_seq.setdefault(col, []).append(act_type)

        for col, seq in col_seq.items():
            for a, b in risky_pairs:
                if a in seq and b in seq and seq.index(a) < seq.index(b):
                    self._add_risk("violations", {
                        "type": "ACTION_ORDERING_CONFLICT",
                        "column": col,
                        "sequence": [a, b],
                    })



    def _rollup_risk_level(self):
        if self.report["violations"]:
            self.report["overall_risk_level"] = "HIGH"
            return

        if any(r.get("level") == "CRITICAL" for r in self.report["leakage_risks"]):
            self.report["overall_risk_level"] = "HIGH"
        elif any(r.get("level") == "WARNING" for r in self.report["leakage_risks"]):
            self.report["overall_risk_level"] = "MEDIUM"
        else:
            self.report["overall_risk_level"] = "LOW"


    def apply_single_action(self, df: pd.DataFrame, action: Dict[str, Any]):
        return df