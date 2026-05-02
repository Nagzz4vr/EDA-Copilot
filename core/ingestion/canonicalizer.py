import pandas as pd
import json
import numpy as np
from datetime import datetime, timezone
import hashlib
import itertools  
from copy import deepcopy

class Canonicalizer:
    def __init__(self,data:dict):
        self.raw_json=self._sanitize_json(data)
        self.json = self._process_canonical_order(self.raw_json)

    def _sanitize_json(self, data=None):
        """Recursively converts NaNs to None and NumPy types to Python types."""
        if isinstance(data, dict):
            return {k: self._sanitize_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_json(x) for x in data]
        elif isinstance(data, (float, np.floating)):
            if np.isnan(data) or np.isinf(data):
                return None
            return float(f"{float(data):.6g}")
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, (bool, np.bool_)):
            return bool(data)

        elif pd.isna(data):
            return None
         
        return data


    def _process_canonical_order(self, data):
        """Ensures stable, sorted structure across all levels."""
        data = deepcopy(data)
        if "columns" in data and isinstance(data["columns"], list):
            data["columns"].sort(key=lambda x: (x.get('type', ''), x.get('name', '')))
            for col in data["columns"]:
                if "flags" in col and isinstance(col["flags"], list):
                    col["flags"].sort()
                for key in ["stats", "signals", "missing"]:
                    if key in col and isinstance(col[key], dict):
                        col[key] = {k: col[key][k] for k in sorted(col[key].keys())}
        return {k: data[k] for k in sorted(data.keys())}
    

    
    def add_state_info(self,schema_version="1.0.0"):

        content_str = json.dumps(self.json, sort_keys=True)


        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:32]


        fingerprint_data = [{"n": c["name"], "t": c["type"]} for c in self.json.get("columns", [])]
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()

        return {
            "metadata": {
                "state_uuid": content_hash,
                "fingerprint": fingerprint,
                "schema_version": schema_version
            },
            "canonical_data": self.json  # Pure, deterministic data
        }
