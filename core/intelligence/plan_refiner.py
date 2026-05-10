from typing import Dict, Any, List, Optional
import collections
import copy
import pandas as pd


class PlanRefiner:
    def __init__(
        self,
        plan: Dict[str, Any],
        impact_report: Dict[str, Any],
        signal_graph: Dict[str, Any],
        max_cols: int = 100,
        memory_threshold_mb: float = 512.0,
        label_column: Optional[str] = None,
        dataset_sample: Optional[pd.DataFrame] = None,
    ):
        self.original_plan = plan
        self.impact_report = impact_report
        self.signal_graph = signal_graph
        self.plan = copy.deepcopy(plan)
        self.max_cols = max_cols
        self.memory_threshold_mb = memory_threshold_mb
        self.label_column = label_column
        self.dataset_sample = dataset_sample
        self.removed_actions: List[Dict] = []

    def refine(self) -> Dict[str, Any]:
        # 1. Normalize actions at entry to prevent schema drift
        self.plan["actions"] = [
            self._normalize_action(a)
            for a in self.plan.get("actions", [])
        ]

        self._apply_constraint_violation_fixes()
        self._apply_dimensionality_explosion_prevention()
        self._apply_memory_optimization()
        self._apply_leakage_prevention()
        self._apply_pipeline_consolidation()

        if self._has_changes():
            self.plan["version"] = self.original_plan.get("version", 1) + 1

        # 2. Sanitize output: Strip keys like `fit_on` that aren't in the strict Pydantic model
        for action in self.plan.get("actions", []):
            action.pop("fit_on", None)

        return self.plan

    # --- Schema Normalization & Safe Access Helpers ---

    def _normalize_action(self, a: Dict) -> Dict:
        return {
        "id":             a.get("id", ""),
        "action_type":    a.get("action_type"),
        "target_columns": a.get("target_columns", []),
        "parameters":     a.get("parameters", {}),
        "metadata":       a.get("metadata", {}),
        "fit_on":         a.get("fit_on", "train_only"),
    }

    def _edge_source(self, e: Dict) -> str:
        return e.get("source_id") or e.get("source")

    def _edge_target(self, e: Dict) -> str:
        return e.get("target_id") or e.get("target")

    # --- Refinement Logic ---

    def _apply_constraint_violation_fixes(self):
        for violation in self.impact_report.get("violations", []):
            offenders = self._find_offending_actions(violation)

            if not offenders:
                continue

            self.plan["actions"] = self._remove_actions(offenders)

            for off in offenders:
                replacements = self._find_replacements(off)

                if replacements:
                    self.plan["actions"].append(
                        {
                            "action_type": replacements[0].get("action_type"),
                            "target_columns": replacements[0].get("target_columns", []),
                        }
                    )

    def _find_offending_actions(self, violation: Dict) -> List[Dict]:
        offenders = []

        if violation.get("type") == "ACTION_ORDERING_CONFLICT":
            col = violation.get("column")
            ops = violation.get("sequence", [])

            for action in self.plan["actions"]:
                if col in action.get("target_columns", []) and action.get("action_type") in ops:
                    offenders.append(action)

        return offenders

    def _remove_actions(self, offenders: List[Dict]) -> List[Dict]:
        offenders_id = {
            (a.get("action_type"), tuple(a.get("target_columns", [])))
            for a in offenders
        }

        return [
            a for a in self.plan["actions"]
            if (a.get("action_type"), tuple(a.get("target_columns", []))) not in offenders_id
        ]

    def _apply_dimensionality_explosion_prevention(self):
        projections = self.impact_report.get("projections", {})
        current_cols = projections.get("final_column_count", 0)

        if current_cols <= self.max_cols:
            return

        one_hot_actions = [
            a for a in self.plan["actions"]
            if a.get("action_type") == "ONE_HOT_ENCODE"
        ]
        if not one_hot_actions:
            return

        trace = projections.get("column_trace", [])
        delta_by_key: Dict[str, int] = {}

        if trace:
            for t in trace:
                act = t.get("action", {})
                if act.get("action_type") == "ONE_HOT_ENCODE":
                    key = self._action_key(act)
                    delta_by_key[key] = t.get("delta", 0)

        def est_expansion(a: Dict) -> int:
            total = 0
            for c in a.get("target_columns", []):
                if self.dataset_sample is not None and c in self.dataset_sample.columns:
                    k = self.dataset_sample[c].nunique(dropna=True)
                    total += max(0, k - 1)
            return total

        ranked = sorted(
            one_hot_actions,
            key=lambda a: delta_by_key.get(self._action_key(a), est_expansion(a)),
            reverse=True,
        )

        for action in ranked:
            if current_cols <= self.max_cols:
                break

            key = self._action_key(action)
            expansion = delta_by_key.get(key, est_expansion(action))
            if expansion <= 0:
                continue

            replacement = self._choose_replacement(action, expansion)
            self._replace_action(action, replacement)

            new_width = self._replacement_width(replacement)
            current_cols = current_cols - expansion + new_width
            self.removed_actions.append(action)

        self.plan["actions"] = self._topological_sort(self.plan["actions"])

    def _apply_memory_optimization(self):
        actual_mb = self.impact_report.get("projections", {}).get(
            "actual_sample_memory_mb", 0.0
        )
        if actual_mb <= self.memory_threshold_mb:
            return
        if self.dataset_sample is None:
            return

        downcast_cols: List[str] = []
        for col in self.dataset_sample.columns:
            series = self.dataset_sample[col]

            if pd.api.types.is_float_dtype(series):
                if series.dropna().between(-3.4e38, 3.4e38).all():
                    downcast_cols.append(col)

            elif pd.api.types.is_integer_dtype(series):
                col_min, col_max = series.min(), series.max()
                if col_min >= -32_768 and col_max <= 32_767:
                    downcast_cols.append(col)

        if not downcast_cols:
            return

        downcast_action = {
            "action_type": "DOWNCAST",
            "target_columns": downcast_cols,
            "fit_on": "train_only",
        }

        last_impute_idx = -1
        first_scale_idx = len(self.plan["actions"])

        for i, action in enumerate(self.plan["actions"]):
            if "IMPUTE" in action.get("action_type", ""):
                last_impute_idx = i
            if "SCALE" in action.get("action_type", "") and i < first_scale_idx:
                first_scale_idx = i

        insert_at = max(last_impute_idx + 1, 0)
        insert_at = min(insert_at, first_scale_idx)
        self.plan["actions"].insert(insert_at, downcast_action)

    def _apply_leakage_prevention(self):
        HIGH_LEVELS = {"CRITICAL", "HIGH"}

        leaky_cols = {
            r["column"]
            for r in self.impact_report.get("leakage_risks", [])
            if r.get("level") in HIGH_LEVELS and "column" in r
        }

        if leaky_cols:
            self.plan["actions"] = [
                a for a in self.plan["actions"]
                if not leaky_cols.intersection(a.get("target_columns", []))
            ]

        leaky_action_keys = {
            (r.get("action"), r.get("column"))
            for r in self.impact_report.get("action_risks", [])
            if r.get("type") == "TRAIN_TEST_LEAKAGE"
        }

        for action in self.plan["actions"]:
            for act_type, col in leaky_action_keys:
                if (
                    action.get("action_type") == act_type
                    and col in action.get("target_columns", [])
                ):
                    action["fit_on"] = "train_only"

    def _apply_pipeline_consolidation(self):
        has_violations = bool(self.impact_report.get("violations"))
        has_high_leakage = any(
            r.get("level") in {"CRITICAL", "HIGH"}
            for r in self.impact_report.get("leakage_risks", [])
        )
        if has_violations or has_high_leakage:
            return

        connected_pairs = {
            (self._edge_source(e), self._edge_target(e))
            for e in self.signal_graph.get("edges", [])
        }

        def are_independent(a: Dict, b: Dict) -> bool:
            id_a = self._action_key(a)
            id_b = self._action_key(b)
            return (
                (id_a, id_b) not in connected_pairs
                and (id_b, id_a) not in connected_pairs
                and not set(a.get("target_columns", [])) & set(b.get("target_columns", []))
            )

        by_type: Dict[str, List[Dict]] = collections.defaultdict(list)
        for action in self.plan["actions"]:
            by_type[action.get("action_type", "UNKNOWN")].append(action)

        merged: List[Dict] = []
        for act_type, group in by_type.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            can_merge = all(
                are_independent(group[i], group[j])
                for i in range(len(group))
                for j in range(i + 1, len(group))
            )

            if can_merge:
                all_cols = []
                seen_cols = set()
                for a in group:
                    for c in a.get("target_columns", []):
                        if c not in seen_cols:
                            all_cols.append(c)
                            seen_cols.add(c)

                consolidated = {k: v for k, v in group[0].items()}
                consolidated["target_columns"] = all_cols
                merged.append(consolidated)
            else:
                merged.extend(group)

        self.plan["actions"] = self._topological_sort(merged)

    def _find_replacements(self, removed_action: Dict) -> List[Dict]:
        candidates = []
        for node in self.signal_graph.get("nodes", []):
            if tuple(node.get("target_columns", [])) != tuple(removed_action.get("target_columns", [])):
                continue
            if node.get("action_type") == removed_action.get("action_type"):
                continue
            if not self._is_compatible(node):
                continue
            candidates.append(node)

        candidates.sort(
            key=lambda n: n.get("benefit_score", 0) - n.get("cost_score", 0),
            reverse=True,
        )
        return candidates[:1]

    def _is_compatible(self, node: Dict) -> bool:
        selected_ids = {
            f"{a.get('action_type')}_{'_'.join(a.get('target_columns', []))}"
            for a in self.plan["actions"]
        }

        node_id = node.get("node_id")

        for e in self.signal_graph.get("edges", []):
            if e.get("edge_type") != "CONFLICTS_WITH":
                continue

            src = self._edge_source(e)
            tgt = self._edge_target(e)

            if src == node_id and tgt in selected_ids:
                return False
            if tgt == node_id and src in selected_ids:
                return False

        return True

    def _has_changes(self) -> bool:
        def normalise(actions: List[Dict]) -> List[tuple]:
            result = []
            for a in actions:
                key = (a.get("action_type"), tuple(sorted(a.get("target_columns", []))))
                result.append(key)
            return sorted(result)

        return normalise(self.plan.get("actions", [])) != normalise(
            self.original_plan.get("actions", [])
        )

    def _action_key(self, action: Dict) -> str:
        return f"{action.get('action_type')}_{'_'.join(action.get('target_columns', []))}"

    def _choose_replacement(self, action: Dict, expansion: int) -> Dict:
        if self.label_column:
            replacement_type = "TARGET_ENCODE"
        else:
            replacement_type = "FEATURE_HASH"

        return {
            "action_type": replacement_type,
            "target_columns": action.get("target_columns", []),
            "fit_on": action.get("fit_on", "train_only"),
            **({"n_components": 16} if replacement_type == "FEATURE_HASH" else {}),
        }

    def _replace_action(self, old: Dict, new: Dict):
        for i, action in enumerate(self.plan["actions"]):
            if self._action_key(action) == self._action_key(old):
                self.plan["actions"][i] = new
                return
        raise ValueError(f"Action not found in plan: {self._action_key(old)}")

    def _replacement_width(self, action: Dict) -> int:
        act_type = action.get("action_type")
        if act_type == "TARGET_ENCODE":
            return len(action.get("target_columns", []))
        if act_type == "FEATURE_HASH":
            return action.get("n_components", 16)

        return len(action.get("target_columns", []))

    def _topological_sort(self, actions: List[Dict]) -> List[Dict]:
        key_to_action = {self._action_key(a): a for a in actions}
        action_keys = set(key_to_action)

        in_degree: Dict[str, int] = {k: 0 for k in action_keys}
        adj: Dict[str, List[str]] = collections.defaultdict(list)

        for edge in self.signal_graph.get("edges", []):
            if edge.get("edge_type") != "PREREQUISITE_FOR":
                continue
            
            src = self._edge_source(edge)
            tgt = self._edge_target(edge)
            
            if src in action_keys and tgt in action_keys:
                adj[src].append(tgt)
                in_degree[tgt] += 1

        queue = collections.deque(
            [k for k in action_keys if in_degree[k] == 0]
        )
        sorted_keys: List[str] = []

        while queue:
            node = queue.popleft()
            sorted_keys.append(node)
            for neighbour in adj[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(sorted_keys) != len(action_keys):
            raise ValueError(
                "Cycle detected in action graph during topological sort"
            )

        return [key_to_action[k] for k in sorted_keys]