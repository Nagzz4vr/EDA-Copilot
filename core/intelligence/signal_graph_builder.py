from typing import List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EdgeType(Enum):
    CONFLICTS_WITH = "CONFLICTS_WITH"
    PREREQUISITE_FOR = "PREREQUISITE_FOR"

@dataclass(frozen=True)
class ActionNode:
    id: str
    action_type: str
    target_columns: tuple
    benefit_score: float
    cost_score: float
    risk_level: RiskLevel
    originating_rule: str

@dataclass
class ActionEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType

class SignalGraphBuilder:
    def __init__(self):
        self.nodes:Dict[str,ActionNode]={}
        self.edges:List[ActionEdge]=[]
        self._risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


    def _build_graph(self,scored_signals: List[Dict[str, Any]]) -> Dict[str, Any]:

        for signal in scored_signals:
            self._create_nodes_from_signal(signal)

        self._map_relationships()
        return {
            "nodes": [
                {
                    "id": n.id,
                    "action_type": n.action_type,
                    "target_columns": list(n.target_columns),
                    "benefit_score": n.benefit_score,
                    "cost_score": n.cost_score,
                    "risk_level": n.risk_level.value,
                    "originating_rule": n.originating_rule
                }
                for n in self.nodes.values()
            ],
          "edges": [
                    {"source_id": e.source_id, "target_id": e.target_id, "edge_type": e.edge_type.value}
                for e in self.edges
            ]
        }
    
    def _create_nodes_from_signal(self,signal: Dict[str, Any]):

        action=signal.get("action")
        cols = tuple(signal.get("details", {}).get("affected_columns", ["dataset"]))

        benefit = signal.get("computed_priority", 0) / 100.0
        severity = signal.get("severity", "info").upper()
        risk = RiskLevel[severity] if severity in RiskLevel.__members__ else RiskLevel.LOW

        if action == "warning" and "missing" in signal.get("rule_name", "").lower():
            self._add_node(cols, "IMPUTE_MEAN", benefit, 0.1, RiskLevel.LOW, signal)
            self._add_node(cols, "IMPUTE_KNN", benefit * 1.2, 0.6, RiskLevel.MEDIUM, signal)

        else:
            cost = self._estimate_cost(action)
            self._add_node(cols, action.upper(), benefit, cost, risk, signal)

    def _add_node(self, cols, act_type, benefit, cost, risk, signal):
        node_id = f"{act_type}_{'_'.join(cols)}"
        rule_name = signal.get("rule_name", "unknown")

        if node_id in self.nodes:
            existing = self.nodes[node_id]
            

            benefit = min(1.0, existing.benefit_score + benefit)
            

            risk = max(
                [existing.risk_level, risk],
                key=lambda r: self._risk_order.index(r.value)
            )
            
            cost = min(existing.cost_score, cost)
            
            rule_name = f"{existing.originating_rule}|{rule_name}"

        self.nodes[node_id] = ActionNode(
                            id=node_id,
                            action_type=act_type,
                            target_columns=cols,
                            benefit_score=benefit,
                            cost_score=cost,
                            risk_level=risk,
                            originating_rule=rule_name
                                        )

    def _estimate_cost(self, action: str) -> float:
        costs = {
            "block_analysis": 1.0,
            "drop": 0.05,
            "one_hot_encode": 0.4,
            "use_feature_hashing": 0.3,
            "route_to_nlp_eda": 0.8
        }
        return costs.get(action, 0.2)
    
    def _map_relationships(self):
        node_list = list(self.nodes.values())
        seen_edges = set() 

        for i, node_a in enumerate(node_list):
            for node_b in node_list[i+1:]:
                if self._is_conflict(node_a, node_b):
                    self._add_edge(node_a.id, node_b.id, EdgeType.CONFLICTS_WITH, True, seen_edges)
                

                self._check_prereq(node_a, node_b, seen_edges)
                self._check_prereq(node_b, node_a, seen_edges)

    def _check_prereq(self, a: ActionNode, b: ActionNode, seen: Set):
        overlap = set(a.target_columns) & set(b.target_columns)
        if not overlap:
            return
        a_type, b_type = a.action_type, b.action_type
        
        is_prereq = (
            ("IMPUTE" in a_type and any(x in b_type for x in ["ENCODE", "HASH", "SCALE", "CREATE"])) or
            ("ENCODE" in a_type and any(x in b_type for x in ["SCALE", "CREATE"])) or
            ("SCALE" in a_type and "CREATE" in b_type)
        )
        if is_prereq:
            self._add_edge(a.node_id, b.node_id, EdgeType.PREREQUISITE_FOR, False, seen)

    def _add_edge(self, src: str, tgt: str, e_type: EdgeType, bidirectional: bool, seen: Set):
        edge_key = (src, tgt, e_type.value)
        if edge_key not in seen:
            self.edges.append(ActionEdge(src, tgt, e_type))
            seen.add(edge_key)
        
        if bidirectional:
            rev_key = (tgt, src, e_type.value)
            if rev_key not in seen:
                self.edges.append(ActionEdge(tgt, src, e_type))
                seen.add(rev_key)

    def _is_conflict(self, a: ActionNode, b: ActionNode) -> bool:
        if not (set(a.target_columns) & set(b.target_columns)):
            return False
        

        a_type, b_type = a.action_type, b.action_type
        

        if "DROP" in (a_type, b_type):
            return True

        if "IMPUTE" in a_type and "IMPUTE" in b_type:
            return True

        if  any(x in a_type for x in ["ENCODE", "HASH"]) and \
            any(x in b_type for x in ["ENCODE", "HASH"]):
            return True

        return False