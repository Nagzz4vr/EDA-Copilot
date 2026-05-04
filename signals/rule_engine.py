import pkgutil
import importlib
import inspect
from typing import List, Type,Dict,Any
from rules.base_rule import BaseRule,RuleOutput

class RuleEngine:
    def __init__(self,context:dict):
        self.context=context
        self.rules= self._discover_rules() 

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
                print(f"Error in rule {rule.name}: {e}")
                continue
        
        return signal_set