from typing import Dict,Any

class PlanTranslator:
    ACTION_MAP = {
        "IMPUTE_MEAN": ("impute", {"strategy": "mean"}),
        "IMPUTE_MEDIAN": ("impute", {"strategy": "median"}),
        "IMPUTE_KNN": ("impute", {"strategy": "knn"}),
        "SCALE_STANDARD": ("scale", {"method": "standard"}),
        "SCALE_MINMAX": ("scale", {"method": "minmax"}),
        "ONE_HOT_ENCODE": ("encode", {"method": "onehot"}),
        "TARGET_ENCODE": ("encode", {"method": "target"}),
        "FEATURE_HASH": ("encode", {"method": "hash"}),
        "DROP": ("drop_column", {}),
        "DOWNCAST": ("cast_dtype", {"dtype": "float32"}),
    }

    def __init__(self, refined_plan, signal_graph):
        self.refined_plan = refined_plan
        self.signal_graph = signal_graph

    def translate(self) -> Dict[str, Any]:
        transforms = []

        node_index = {}
        if not self.refined_plan:
            raise ValueError("refined_plan is None")

        actions = self.refined_plan.get("actions")
        if not actions:
            raise ValueError("No actions in refined_plan")

        for action in actions:
            transforms.extend(
                self._explode_action(action, node_index)
            )
        self._attach_dependencies(
            transforms=transforms,
            node_index=node_index,
        )

        return {
            "transforms": [
                self._serialize_transform(t)
                for t in transforms
            ]
        }
    
    def _explode_action(self, action, node_index):

        action_type = action["action_type"]

        if action_type not in self.ACTION_MAP:
            return []

        operation, params = self.ACTION_MAP[action_type]
        transforms = []

        for column in action.get("target_columns", []):

            transform_id = f"{action['id']}_{column}"   
                                                        


            transform = {
                "id":         transform_id,
                "operation":  operation,
                "column":     column,
                "params":     params.copy(),
                "depends_on": []
            }

            transforms.append(transform)
            node_index[transform_id] = transform         # fix 2: key matches transform id

        return transforms

    def _attach_dependencies(self, transforms, node_index):

        # Placeholder for future DAG dependency logic
        return

    def _serialize_transform(self, transform):

        return transform