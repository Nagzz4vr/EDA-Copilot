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

ACTION_STEP_MAP: Dict[str, Tuple[str, str, Dict]] = {

    "impute_mean":         ("sklearn.impute",             "SimpleImputer",         {"strategy": "mean"}),
    "impute_median":       ("sklearn.impute",             "SimpleImputer",         {"strategy": "median"}),
    "impute_mode":         ("sklearn.impute",             "SimpleImputer",         {"strategy": "most_frequent"}),
    "impute_constant":     ("sklearn.impute",             "SimpleImputer",         {"strategy": "constant"}),
    "impute_knn":          ("sklearn.impute",             "KNNImputer",            {"n_neighbors": 5}),

    "scale_standard":      ("sklearn.preprocessing",      "StandardScaler",        {}),
    "scale_minmax":        ("sklearn.preprocessing",      "MinMaxScaler",          {}),
    "scale_robust":        ("sklearn.preprocessing",      "RobustScaler",          {}),
    "normalize":           ("sklearn.preprocessing",      "Normalizer",            {}),

    "encode_onehot":       ("sklearn.preprocessing",      "OneHotEncoder",         {"handle_unknown": "ignore", "sparse_output": False}),
    "encode_ordinal":      ("sklearn.preprocessing",      "OrdinalEncoder",        {}),
    "encode_label":        ("sklearn.preprocessing",      "LabelEncoder",          {}),
    "encode_target":       ("category_encoders",          "TargetEncoder",         {}),

    "log_transform":       ("sklearn.preprocessing",      "FunctionTransformer",   {"func": "numpy.log1p"}),
    "sqrt_transform":      ("sklearn.preprocessing",      "FunctionTransformer",   {"func": "numpy.sqrt"}),
    "power_transform":     ("sklearn.preprocessing",      "PowerTransformer",      {"method": "yeo-johnson"}),
    "quantile_transform":  ("sklearn.preprocessing",      "QuantileTransformer",   {"output_distribution": "normal"}),
    "binarize":            ("sklearn.preprocessing",      "Binarizer",             {}),

    "drop_column":         ("sklearn.feature_selection",  "VarianceThreshold",     {"threshold": 0.0}),
    "select_kbest":        ("sklearn.feature_selection",  "SelectKBest",           {}),

    "clip_outliers":       ("sklearn.preprocessing",      "FunctionTransformer",   {}),
}

@dataclass
class ExportManifest:
    job_id:        str
    pkl_path:      str
    script_path:   str
    steps:         List[str]       
    n_steps:       int
    exported_at:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        return {
            "job_id":      self.job_id,
            "pkl_path":    self.pkl_path,
            "script_path": self.script_path,
            "steps":       self.steps,
            "n_steps":     self.n_steps,
            "exported_at": self.exported_at,
        }
    
class PipelineExporter:
    def __init__(self,output_dir: str = "outputs/pipelines",step_map_overrides: Optional[Dict] = None,):
        self._output_dir = Path(output_dir)
        self._step_map   = {**ACTION_STEP_MAP, **(step_map_overrides or {})}

    def export(self, plan: Any, job_id: str) -> ExportManifest:
        actions   = self._extract_actions(plan)
        steps     = self._build_steps(actions)
        pipeline  = Pipeline(steps)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        pkl_path    = self._output_dir / f"{job_id}_pipeline.pkl"
        script_path = self._output_dir / f"{job_id}_pipeline.py"
        joblib.dump(pipeline, pkl_path)
        self._write_script(steps, script_path, job_id)
        step_names = [name for name, _ in steps]
        return ExportManifest(
            job_id      = job_id,
            pkl_path    = str(pkl_path),
            script_path = str(script_path),
            steps       = step_names,
            n_steps     = len(step_names),
        )
    
    def _extract_actions(self, plan: Any) -> List[Dict]:
        if hasattr(plan, "actions"):
            raw = plan.actions
        elif isinstance(plan, dict):
            raw = plan.get("actions", [])
        else:
            raise TypeError(f"Cannot extract actions from plan type {type(plan).__name__}")
        if not isinstance(raw, list):
            raise ValueError("plan.actions must be a list")
        return raw
    
    def _build_steps(self, actions: List[Dict]) -> List[Tuple[str, Any]]:
        steps       = []
        seen_names  = {}  
        for action in actions:
            action_type = action.get("action_type", "")
            column      = action.get("column", "")
            params      = action.get("params", {})
            if action_type not in self._step_map:
                continue
            module_path, class_name, default_params = self._step_map[action_type]
            transformer = self._instantiate(module_path, class_name, default_params, params)
            base_name = f"{action_type}_{column}" if column else action_type
            suffix    = seen_names.get(base_name, 0)
            step_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
            seen_names[base_name] = suffix + 1
            steps.append((step_name, transformer))
        return steps
    
    @staticmethod
    def _instantiate(module_path: str,class_name: str,default_params: Dict,override_params: Dict,) -> Any:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"PipelineExporter: cannot load {module_path}.{class_name}: {exc}"
            ) from exc
        
        valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        merged     = {**default_params, **override_params}
        filtered   = {k: v for k, v in merged.items() if k in valid_keys}
        return cls(**filtered)
    
    def _write_script(self,steps: List[Tuple[str, Any]],path: Path,job_id: str,) -> None:
        imports: Dict[str, List[str]] = {}
        step_lines: List[str] = []
        for step_name, transformer in steps:
            module = transformer.__class__.__module__
            cls    = transformer.__class__.__name__
            imports.setdefault(module, [])
            if cls not in imports[module]:
                imports[module].append(cls)
 
            params_repr = ", ".join(
                f"{k}={v!r}"
                for k, v in transformer.get_params().items()
                if v is not None
            )
            step_lines.append(f'    ("{step_name}", {cls}({params_repr})),')

        import_block = "\n".join(
            f"from {mod} import {', '.join(sorted(clses))}"
            for mod, clses in sorted(imports.items())
        )

        steps_block = "\n".join(step_lines)

        script = textwrap.dedent(f"""\
            # Auto-generated pipeline for job: {job_id}
            # Generated at: {datetime.now(timezone.utc).isoformat()}
            # Do not edit manually — re-run the pipeline to regenerate.

            from sklearn.pipeline import Pipeline
            {import_block}

            pipeline = Pipeline(steps=[
            {steps_block}
            ])

            # Usage:
            #   pipeline.fit(X_train, y_train)
            #   X_transformed = pipeline.transform(X_test)
        """)

        path.write_text(script, encoding="utf-8")

