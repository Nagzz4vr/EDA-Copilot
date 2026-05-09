import pkgutil
import importlib
import inspect
from typing import List, Type,Dict,Any
from signals.rules.base_rule import BaseRule, RuleOutput
import traceback

class RuleEngine:
    def __init__(self,context:dict,logger=None):
        self.context=context
        self.rules= self._discover_rules() 
        self.logger=logger

    def _discover_rules(self)->list[BaseRule]:
        rules=[]
        package = importlib.import_module("signals.rules")

        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            module = importlib.import_module(module_name)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseRule) and obj is not BaseRule:
                    rules.append(obj())
        rules.sort(key=lambda rule: getattr(rule, "priority", 0), reverse=True)

        return rules

    # def _discover_rules(self) -> list[BaseRule]:
    #     rules = []

    #     package = importlib.import_module("signals.rules")

    #     print("PACKAGE:", package)
    #     print("PATH:", package.__path__)

    #     for _, module_name, _ in pkgutil.walk_packages(
    #         package.__path__,
    #         package.__name__ + "."
    #     ):

    #         print(f"\nIMPORTING MODULE: {module_name}")

    #         try:
    #             module = importlib.import_module(module_name)

    #             print("SUCCESS")

    #             for name, obj in inspect.getmembers(module, inspect.isclass):

    #                 print("FOUND CLASS:", name)

    #                 try:
    #                     print("ISSUBCLASS:", issubclass(obj, BaseRule))
    #                 except Exception as e:
    #                     print("ISSUBCLASS ERROR:", e)

    #                 if issubclass(obj, BaseRule) and obj is not BaseRule:
    #                     print("ADDING RULE:", name)
    #                     rules.append(obj())

    #         except Exception as e:
    #             print("MODULE IMPORT FAILED:", module_name)
    #             traceback.print_exc()

    #     rules.sort(
    #         key=lambda rule: getattr(rule, "priority", 0),
    #         reverse=True
    #     )

    #     print("\nFINAL RULE COUNT:", len(rules))

    #     return rules

    def run(self) -> List[Dict[str, Any]]:
        signal_set = []
        
        for rule in self.rules:
            try:
                if rule.applies(self.context):
                    result = rule.run(self.context)
                    
                    if not isinstance(result, RuleOutput):
                        raise TypeError(
                            f"Rule {rule.name} must return RuleOutput, got {type(result)}"
                        )
                    
                    signal_set.append(result.to_dict())
                    
            except Exception as e:
                self.logger.log(
                    tool="RULE_ENGINE",
                    intent="Rule execution failed",
                    inputs={
                        "rule": rule.name,
                        "rule_class": rule.__class__.__name__,
                    },
                    outputs={
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    confidence=0.0,
                )

                print("\n" + "=" * 80)
                print(f"RULE FAILURE: {rule.name}")
                print("=" * 80)
                print(traceback.format_exc())

        return signal_set
    
if __name__ == "__main__":
    re=RuleEngine()