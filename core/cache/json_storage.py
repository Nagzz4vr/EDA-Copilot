import os
import json
from typing import Dict, Any


import os
import json
import tempfile
from typing import Dict, Any

class JsonStorage:

    def __init__(self, cache_dir="cache_storage"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str):
        return os.path.join(self.cache_dir, f"{key}.json")

    def write(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path(key)

        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, default=str)

            os.replace(tmp_path, path)  # atomic commit

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def read(self, key: str):
        path = self._path(key)
    
        if not os.path.exists(path):
            return None
    
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    
        except json.JSONDecodeError:
            # purge corrupted cache
            os.remove(path)
            return None