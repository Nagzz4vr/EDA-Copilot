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

        ordered_nodes = DAGBuilder._topological_sort(nodes)
        return DAG(nodes=ordered_nodes,version=plan.get("version", "1.0"))
    
    @staticmethod
    def _topological_sort(nodes):

        node_map = {n.id: n for n in nodes}

        in_degree = {n.id: 0 for n in nodes}
        adjacency = {n.id: [] for n in nodes}

        for node in nodes:
            for dep in node.depends_on:
                adjacency[dep].append(node.id)
                in_degree[node.id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]

        ordered = []

        while queue:

            current = queue.pop(0)

            ordered.append(node_map[current])

            for neighbor in adjacency[current]:

                in_degree[neighbor] -= 1

                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(nodes):
            raise ValueError("Cycle detected in DAG")

        return ordered