from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import collections
from core.schema.shared_shema import *


RISK_MAP = {
    "LOW": 0.0,
    "MEDIUM": 0.3,
    "HIGH": 0.7,
    "CRITICAL": 1.0
}
class PlanOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.lambda1 = config.get("lambda_cost", 0.3)
        self.lambda2 = config.get("lambda_risk", 0.5)
        self.lambda3 = config.get("lambda_length", 0.1)
        self.budget = config.get("budget", 5.0)
        self.max_cols = config.get("max_columns", 100)
        self.allow_critical = config.get("allow_critical_risk", False)

    def optimize(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        nodes = graph["nodes"]
        edges = graph["edges"]

        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise TypeError("Graph data must contain list 'nodes' and 'edges'")

        selected_nodes = []
        current_cost = 0.0
        remaining = nodes.copy()

        while remaining:
            scored = sorted(
                remaining,
                key=lambda n: self._score(n, selected_nodes),
                reverse=True
            )

            picked = None

            for node in scored:
                if self._is_feasible(node, selected_nodes, edges, current_cost):
                    picked = node
                    break

            if picked is None:
                break

            selected_nodes.append(picked)

            current_cost += picked.get("cost_score", 0.0)

            remaining = [n for n in remaining if n["id"] != picked["id"]]  

        ordered_actions = self._topological_sort(selected_nodes, edges)

        self._validate_invariants(ordered_actions)

        selected_ids = {n["id"] for n in selected_nodes}  
        speculative = self._get_speculative_candidates(nodes, selected_ids)

        return self._build_plan(ordered_actions, current_cost, speculative)

    def _score(self, node: Dict[str, Any], selected_nodes: List[Dict[str, Any]]):
        risk_penalty = RISK_MAP.get(node.get("risk_level", "LOW"), 0.0)
        current_risk = self._aggregate_risk(selected_nodes)

        return (
            node.get("benefit_score", 0.0)
            - self.lambda1 * node.get("cost_score", 0.0)   
            - self.lambda2 * max(current_risk, risk_penalty)
            - self.lambda3 * len(selected_nodes)
        )

    def _topological_sort(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        node_ids = {n["id"] for n in nodes}  

        relevant_edges = [
            e for e in edges
            if e["source_id"] in node_ids              
            and  e["target_id"] in node_ids            
            and e["edge_type"] == "PREREQUISITE_FOR"
        ]

        in_degree = {n_id: 0 for n_id in node_ids}
        adj = collections.defaultdict(list)

        for e in relevant_edges:
            adj[e["source_id"]].append( e["target_id"])   
            in_degree[ e["target_id"]] += 1            

        queue = collections.deque([n for n in node_ids if in_degree[n] == 0])
        sorted_ids = []

        while queue:
            u = queue.popleft()
            sorted_ids.append(u)

            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_ids) != len(node_ids):
            raise ValueError("Cycle detected in action graph")

        node_lookup = {n["id"]: n for n in nodes}  
        return [node_lookup[n_id] for n_id in sorted_ids]

    def _get_speculative_candidates(self, nodes: List[Dict], selected_ids: Set[str]) -> List[Dict]:
        return [
            n for n in nodes
            if n["id"] not in selected_ids and n["benefit_score"] > 0.7  
        ]

    def _aggregate_risk(self, selected_nodes):
        risk_values = [RISK_MAP.get(n["risk_level"], 0.0) for n in selected_nodes]
        return max(risk_values) if risk_values else 0.0

    def _is_feasible(self, node, selected_nodes, edges, current_cost):
        if current_cost + node.get("cost_score", 0.0) > self.budget:  
            return False

        if node.get("risk_level") == "CRITICAL" and not self.allow_critical:
            return False

        selected_ids = {n["id"] for n in selected_nodes}  

        for e in edges:
            if e.get("edge_type") != "CONFLICTS_WITH":
                continue

            src = e.get("source_id") or e.get("source")   # ← fix
            tgt = e.get("target_id") or e.get("target")   # ← fix

            if (src == node["id"] and tgt in selected_ids) or \
               (tgt == node["id"] and src in selected_ids):
                return False

        return True

    def _validate_invariants(self, ordered_nodes):
        seen = set()
        for n in ordered_nodes:
            key = (n["action_type"], tuple(n["target_columns"]))
            if key in seen:
                raise ValueError("Duplicate action detected")
            seen.add(key)

    def _build_plan(self, ordered_nodes, total_cost, speculative_candidates):
        total_benefit = sum(n["benefit_score"] for n in ordered_nodes)

        risk_priority = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        max_risk = (
            max(
                ordered_nodes,
                key=lambda n: risk_priority.get(n["risk_level"], 0)
            )["risk_level"]
            if ordered_nodes else "LOW"
        )

        metadata = {
            "total_benefit": total_benefit,
            "total_cost": total_cost,
            "max_risk": max_risk,
            "estimated_memory_delta_mb": 0.0
        }

    
        actions = [
            {
                "id": n["id"],
                "action_type": n["action_type"],
                "target_columns": n.get("target_columns", []),

                # execution-safe container only
                "parameters": {
                    "benefit_score": n.get("benefit_score", 0.0),
                    "cost_score": n.get("cost_score", 0.0),
                    "risk_level": n.get("risk_level", "LOW"),
                    "originating_rule": n.get("originating_rule", "")
                },

                "metadata": {}
            }
            for n in ordered_nodes
        ]

        return ExecutionPlan(
            version=1,
            actions=actions,
            metadata=metadata,
            speculative_candidates=speculative_candidates
        )