from dataclasses import dataclass, field
from typing import List, Dict, Any

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


MINIMUM_ACCEPTABLE_DELTA = -0.02


@dataclass
class ValidationReport:
    passed: bool
    delta: float
    score_before: float
    score_after: float
    violations: List[Dict[str, Any]] = field(default_factory=list)


class ValidationEngine:

    def __init__(self, sample_tester, logger):
        self.sample_tester = sample_tester
        self.logger = logger

    def run(self, df_before: pd.DataFrame, target_column: str, transforms: list) -> ValidationReport:
        X_before = df_before.drop(columns=[target_column])
        y_before = df_before[target_column]
        (X_train_before, X_test_before, y_train_before, y_test_before) = train_test_split(X_before, y_before, test_size=0.2, random_state=42)

        baseline_model = DummyClassifier(strategy="most_frequent")
        baseline_model.fit(X_train_before, y_train_before)
        preds_before = baseline_model.predict(X_test_before)
        score_before = accuracy_score(y_test_before, preds_before)

        df_after = self.sample_tester.apply(df_before.copy(), transforms)

        X_after = df_after.drop(columns=[target_column])
        y_after = df_after[target_column]

        (X_train_after, X_test_after, y_train_after, y_test_after) = train_test_split(X_after, y_after, test_size=0.2, random_state=42)

        after_model = DummyClassifier(strategy="most_frequent")
        after_model.fit(X_train_after, y_train_after)
        preds_after = after_model.predict(X_test_after)
        score_after = accuracy_score(y_test_after, preds_after)

        delta = score_after - score_before
        violations = []

        if delta < MINIMUM_ACCEPTABLE_DELTA:
            violations.append({
                "severity": "HIGH",
                "message": f"Model performance degraded by {delta:.4f}"
            })

        if self.sample_tester.has_data_loss(df_before, df_after):
            violations.append({"severity": "HIGH", "message": "Potential data loss detected"})

        passed = len(violations) == 0
        report = ValidationReport(passed=passed, delta=delta, score_before=score_before, score_after=score_after, violations=violations)
        
        self.logger.log(
            tool="VALIDATION_ENGINE",
            intent="Validation completed",
            inputs={"num_transforms": len(transforms)},
            outputs={"passed": passed, "delta": delta},
            confidence=0.95
        )
        return report