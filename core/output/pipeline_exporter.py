from __future__ import annotations

import importlib
import inspect
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline
import numpy as np


ACTION_STEP_MAP: Dict[str, Tuple[str, str, Dict]] = {

    "impute_mean":     ("sklearn.impute", "SimpleImputer", {"strategy": "mean"}),
    "impute_median":   ("sklearn.impute", "SimpleImputer", {"strategy": "median"}),
    "impute_mode":     ("sklearn.impute", "SimpleImputer", {"strategy": "most_frequent"}),
    "impute_constant": ("sklearn.impute", "SimpleImputer", {"strategy": "constant"}),
    "impute_knn":      ("sklearn.impute", "KNNImputer", {"n_neighbors": 5}),

    "one_hot_encode":  ("sklearn.preprocessing", "OneHotEncoder", {"handle_unknown": "ignore", "sparse_output": False}),
    "label_encode":    ("sklearn.preprocessing", "OrdinalEncoder", {}),
    "ordinal_encode":  ("sklearn.preprocessing", "OrdinalEncoder", {}),

    "scale_standard":  ("sklearn.preprocessing", "StandardScaler", {}),
    "scale_minmax":    ("sklearn.preprocessing", "MinMaxScaler", {}),
    "scale_robust":    ("sklearn.preprocessing", "RobustScaler", {}),

    "drop_column":     ("sklearn.feature_selection", "VarianceThreshold", {"threshold": 0.0}),
}


@dataclass
class ExportManifest:
    job_id: str
    pkl_path: str
    script_path: str
    steps: List[str]
    n_steps: int
    exported_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "pkl_path": self.pkl_path,
            "script_path": self.script_path,
            "steps": self.steps,
            "n_steps": self.n_steps,
            "exported_at": self.exported_at,
        }


class PipelineExporter:

    def __init__(self, output_dir: str = "outputs/pipelines", step_map_overrides: Optional[Dict] = None):
        self._output_dir = Path(output_dir)
        self._step_map = {**ACTION_STEP_MAP, **(step_map_overrides or {})}

    # ---------------- CORE EXPORT ---------------- #

    def export(self, plan: Any, job_id: str) -> ExportManifest:

        actions = self._extract_actions(plan)
        steps = self._build_steps(actions)

        pipeline = Pipeline(steps)

        self._output_dir.mkdir(parents=True, exist_ok=True)

        pkl_path = self._output_dir / f"{job_id}_pipeline.pkl"
        script_path = self._output_dir / f"{job_id}_pipeline.py"

        joblib.dump(pipeline, pkl_path)
        print("steps")
        print(steps)
        print("script_path")
        print(script_path)
        print("job id")
        print(job_id)
        self._write_script(steps, script_path, job_id)

        step_names = [name for name, _ in steps]

        return ExportManifest(
            job_id=job_id,
            pkl_path=str(pkl_path),
            script_path=str(script_path),
            steps=step_names,
            n_steps=len(step_names),
        )

    # ---------------- ACTION PARSING ---------------- #

    def _extract_actions(self, plan: Any) -> List[Dict]:

        if hasattr(plan, "actions"):
            raw = plan.actions
        elif isinstance(plan, dict):
            raw = plan.get("actions", [])
        else:
            raise TypeError(f"Cannot extract actions from {type(plan)}")

        if not isinstance(raw, list):
            raise ValueError("plan.actions must be list")

        return raw

    # ---------------- STEP BUILDER ---------------- #

    def _build_steps(self, actions: List[Dict]) -> List[Tuple[str, Any]]:

        steps = []
        seen = {}

        for action in actions:

            action_type = action.get("action_type", "").lower()
            columns = action.get("target_columns") or [action.get("column")]
            params = action.get("params", {})

            if action_type not in self._step_map:
                continue

            module_path, class_name, default_params = self._step_map[action_type]

            transformer = self._instantiate(
                module_path,
                class_name,
                default_params,
                params
            )

            # IMPORTANT FIX: one step per column (no multi-column transformers)
            for col in columns:
                if not col:
                    continue

                base_name = f"{action_type}_{col}"
                suffix = seen.get(base_name, 0)
                step_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
                seen[base_name] = suffix + 1

                steps.append((step_name, transformer))

        return steps

    # ---------------- SAFE INSTANTIATION ---------------- #

    @staticmethod
    def _instantiate(module_path: str, class_name: str,
                     default_params: Dict, override_params: Dict):

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}

        merged = {**default_params, **override_params}
        filtered = {k: v for k, v in merged.items() if k in valid_keys}

        return cls(**filtered)


    def _write_script(self, steps, path: Path, job_id: str):

        if not steps:
            raise ValueError("Cannot write pipeline script: steps list is empty")

        imports = {}
        step_lines = []

        for name, transformer in steps:
            module = transformer.__class__.__module__
            cls = transformer.__class__.__name__


            imports.setdefault(module, set())
            imports[module].add(cls)

            # Extract parameters with proper formatting
            param_parts = []
            for k, v in transformer.get_params().items():
                if v is None:
                    continue

                if isinstance(v, type):

                    if v.__module__ == 'numpy':
                        param_parts.append(f"{k}=np.{v.__name__}")
                    else:
                        param_parts.append(f"{k}={v.__name__}")
                elif isinstance(v, float) and np.isnan(v):
                    # Convert nan to np.nan
                    param_parts.append(f"{k}=np.nan")
                else:
                    param_parts.append(f"{k}={v!r}")

            params = ", ".join(param_parts)

            step_lines.append(f'("{name}", {cls}({params}))')


        needs_numpy = any(
            any(
                isinstance(v, type) and v.__module__ == 'numpy'
                or isinstance(v, float) and np.isnan(v)
                for v in transformer.get_params().values()
            )
            for _, transformer in steps
        )


        import_lines = []
        if needs_numpy:
            import_lines.append("import numpy as np")

        import_lines.extend(
            f"from {m} import {','.join(sorted(c))}"
            for m, c in sorted(imports.items())
        )

        import_block = "\n".join(import_lines)

        # Generate steps block with proper indentation (4 spaces for list items)
        steps_block = ",\n    ".join(step_lines)

        # Construct final script - NO leading spaces on any line
        script = f"""\n# Auto-generated pipeline: {job_id}\n# Generated at: {datetime.now(timezone.utc).isoformat()}\nfrom sklearn.pipeline import Pipeline\n{import_block}\npipeline = Pipeline(steps=[\n{steps_block}\n])"""

        # Write to file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script, encoding="utf-8")