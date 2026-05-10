import pkgutil
import importlib
import inspect
from typing import List, Dict, Any
from signals.rules.base_rule import BaseRule, RuleOutput
import traceback


class RuleEngine:
    def __init__(self, context: dict, logger=None):
        self.context = context
        self.logger = logger
        self.rules = self._discover_rules()

    def _discover_rules(self) -> list[BaseRule]:
        seen_classes = set()
        rules = []

        package = importlib.import_module("signals.rules")

        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"[RuleEngine] Failed to import module {module_name}: {e}")
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseRule)
                    and obj is not BaseRule
                    and obj not in seen_classes
                ):
                    seen_classes.add(obj)
                    rules.append(obj())

        rules.sort(key=lambda rule: getattr(rule, "priority", 0), reverse=True)
        print(f"[RuleEngine] Discovered {len(rules)} unique rules.")
        return rules

    def _log_error(self, rule_name: str, stage: str, e: Exception):
        tb = traceback.format_exc()

        if self.logger is not None:
            try:
                self.logger.log(
                    tool="RULE_ENGINE",
                    intent=f"Rule {stage} failed",
                    inputs={
                        "rule": rule_name,
                        "stage": stage,
                    },
                    outputs={
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "traceback": tb,
                    },
                    confidence=0.0,
                )
            except Exception as log_err:
                print(f"[RuleEngine] Logger itself failed: {log_err}")

        print("\n" + "=" * 80)
        print(f"RULE FAILURE [{stage.upper()}]: {rule_name}")
        print("=" * 80)
        print(tb)

    def run(self) -> List[Dict[str, Any]]:
        signal_set = []

        for rule in self.rules:

            # ── Gate 1: applies() ─────────────────────────────────────────
            try:
                should_run = rule.applies(self.context)
            except Exception as e:
                self._log_error(rule.name, "applies", e)
                continue

            if not should_run:
                continue

            # ── Gate 2: run() ─────────────────────────────────────────────
            try:
                result = rule.run(self.context)

                if not isinstance(result, RuleOutput):
                    raise TypeError(
                        f"Rule {rule.name} must return RuleOutput, got {type(result)}"
                    )

                signal_set.append(result.to_dict())

            except Exception as e:
                self._log_error(rule.name, "run", e)
                continue

        print(f"[RuleEngine] {len(signal_set)} signals emitted.")
        return signal_set