import pandas as pd
from typing import Dict, Any
from validation.leakage import LeakageDetector

LEAKY_GLOBAL_OPS = ["IMPUTE_MEAN", "IMPUTE_MEDIAN", "SCALE_STANDARD", "SCALE_MINMAX"]
LEAKAGE_THRESHOLD_CRITICAL = 0.95
LEAKAGE_THRESHOLD_WARNING = 0.85


class GlobalStateSimulator:
    def __init__(self, plan: Dict[str, Any], dataset_sample: pd.DataFrame, dataset_schema: Dict[str, Any],max_col:int=50):
        self.plan = plan
        self.dataset_sample = dataset_sample
        self.dataset_schema = dataset_schema
        self.simulated_df = self.dataset_sample.copy()
        self.report = {
            "plan_invalid": False,
            "violations": [],
            "projections": {},
            "leakage_risks": [],
            "action_risks": [],
            "overall_risk_level": "LOW"
        }
        self.max_col=max_col

    def _add_risk(self, bucket: str, entry: dict):
        self.report[bucket].append(entry)

    def simulate(self) -> Dict[str, Any]:
        self._run_metadata_projection()
        self._run_empirical_sampling()

        if self.report["plan_invalid"]:
            self.report["overall_risk_level"] = "HIGH"
            return self.report

        self._run_leakage_and_risk_analysis()
        self._rollup_risk_level()

        return self.report

    def _rollup_risk_level(self):
        high_levels = {"CRITICAL", "HIGH"}

        has_high_leakage = any(
            r["level"] in high_levels for r in self.report["leakage_risks"]
        )
        has_violations = len(self.report["violations"]) > 0
        
        has_warnings = any(
                r.get("level") == "WARNING"
                for r in self.report["leakage_risks"] + self.report["action_risks"]
                )

        if has_high_leakage or has_violations:
            self.report["overall_risk_level"] = "HIGH"
        elif has_warnings:
            self.report["overall_risk_level"] = "MEDIUM"
        else:
            self.report["overall_risk_level"] = "LOW"

    def _run_metadata_projection(self):
        current_count = self.dataset_schema["dataset_overview"]["num_columns"]

        for action in self.plan["actions"]:
            act_type = action["action_type"]
            targets = action["target_columns"]
            if str(act_type).upper() == "DROP":
                current_count -= len(targets)

        self.report["projections"]["final_column_count"] = current_count

        num_rows = self.dataset_schema["dataset_overview"]["num_rows"]
        bytes_estimate = num_rows * current_count * 8
        total_mem = bytes_estimate / (1024 * 1024)
        self.report["projections"]["memory_footprint_mb"] = total_mem

    def _run_empirical_sampling(self):
        last_action = None

        try:
            for action in self.plan["actions"]:
                last_action = action
                self.simulated_df = self.apply_single_action(self.simulated_df, action)

            

            output_mem_bytes = self.simulated_df.memory_usage(deep=True).sum()
            output_mem_mb = output_mem_bytes / (1024 * 1024)
            self.report["projections"]["actual_sample_memory_mb"] = output_mem_mb

        except Exception as e:
            self.report["plan_invalid"] = True
            self._add_risk("violations", {
                "type": "EMPIRICAL_EXECUTION_FAILED",
                "message": f"Plan failed on sample data: {e}",
                "offending_action": last_action
            })

    def _run_leakage_and_risk_analysis(self):
        label_col = self.dataset_schema.get("label_column")

        if label_col is None:
            self._add_risk("action_risks", {
                "type": "SKIPPED_LEAKAGE_CHECK",
                "reason": "No label_column defined in schema — correlation checks skipped"
            })
        else:
            self._run_target_leakage_check(label_col)

        self._run_temporal_leakage_check()
        self._run_action_ordering_check()

    def _run_target_leakage_check(self, label_col: str):
        for action in self.plan["actions"]:
            if action["action_type"] not in ["ENCODE", "IMPUTE", "SCALE", "CREATE_FEATURE"]:
                continue

            for target_col in action["target_columns"]:
                if target_col not in self.simulated_df.columns:
                    continue

                score = LeakageDetector.calculate_appropriate_metric(
                    self.simulated_df[target_col],
                    self.simulated_df[label_col]
                )

                if abs(score) > LEAKAGE_THRESHOLD_CRITICAL:
                    self._add_risk("leakage_risks", {
                        "column": target_col,
                        "level": "CRITICAL",
                        "reason": "High correlation with label",
                        "score": round(score, 4)
                    })
                elif abs(score) > LEAKAGE_THRESHOLD_WARNING:
                    self._add_risk("leakage_risks", {
                        "column": target_col,
                        "level": "WARNING",
                        "reason": "Suspicious correlation with label",
                        "score": round(score, 4)
                    })

                if LeakageDetector.matches_blacklist(target_col):
                    self._add_risk("leakage_risks", {
                        "column": target_col,
                        "level": "HIGH",
                        "reason": "Target/ID keyword match in column name"
                    })

    def _run_temporal_leakage_check(self):
        for action in self.plan["actions"]:
            if action["action_type"] not in LEAKY_GLOBAL_OPS:
                continue

            if action.get("fit_on") != "train_only":
                for target_col in action["target_columns"]:
                    self._add_risk("action_risks", {
                        "type": "TRAIN_TEST_LEAKAGE",
                        "column": target_col,
                        "action": action["action_type"],
                        "message": "Global stat fitted on full dataset — will leak into test fold"
                    })

    def _run_action_ordering_check(self):
        RISKY_PAIRS = [
            ("IMPUTE", "DROP"),
            ("ENCODE", "SCALE"),
        ]

        # Build per-column ordered action list
        col_action_sequence: Dict[str, list] = {}
        for action in self.plan["actions"]:
            for col in action["target_columns"]:
                col_action_sequence.setdefault(col, []).append(action["action_type"])

        for col, sequence in col_action_sequence.items():
            for op_a, op_b in RISKY_PAIRS:
                if op_a in sequence and op_b in sequence:
                    if sequence.index(op_a) < sequence.index(op_b):
                        self._add_risk("violations", {
                            "type": "ACTION_ORDERING_CONFLICT",
                            "column": col,
                            "sequence": [op_a, op_b],
                            "severity": "HIGH",
                            "message": f"{op_a} before {op_b} on same column is invalid"
                        })

    def apply_single_action(self, df: pd.DataFrame, action: Dict[str, Any]) -> pd.DataFrame:
        # Stub — DAG executor will own this
        return df