from core.intelligence.plan_optimizer import PlanOptimizer
from typing import Dict,Any,Literal,List
from dataclasses import dataclass, field

@dataclass
class DAGNode:
    id: str
    operation: str
    column: str
    params: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)

@dataclass
class DAG:
    nodes: List[DAGNode]
    version: str

class DAGBuilder:

    @staticmethod
    def build(plan: dict) -> DAG:
        if "transforms" not in plan:
            raise ValueError("Plan missing 'transforms'")
        nodes = []
        for transform in plan["transforms"]:
            node = DAGNode(
                id         = transform["id"],
                operation  = transform["operation"], 
                column     = transform["column"],
                params     = transform["params"],
                depends_on = transform.get("depends_on", [])
            )
            nodes.append(node)

        ordered_nodes = (PlanOptimizer._topological_sort(nodes))
        return DAG(nodes=ordered_nodes,version=plan.get("version", "1.0"))