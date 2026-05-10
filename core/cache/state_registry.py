from typing import Dict, Any, List
import os

class StateRegistry:

    def __init__(self, storage):

        self.storage = storage

    def find_one(self, query: Dict[str, Any]):

        state_uuid = query.get("state_uuid")
        if not state_uuid:
            return None
        return self.storage.read(state_uuid)

    def find_many(self,query: Dict[str, Any],sort_by: str = None) -> List[Dict[str, Any]]:
        results = []
        cache_dir = self.storage.cache_dir
        for filename in os.listdir(cache_dir):
            if not filename.endswith(".json"):
                continue
            key = filename.replace(".json", "")
            try:
                entry = self.storage.read(key)
            except Exception:
                continue
            if entry is None:
                continue
            match = True
            for qk, qv in query.items():
                if entry.get(qk) != qv:
                    match = False
                    break
            if match:
                results.append(entry)
        if sort_by:
            results.sort(
                key=lambda x: float(x.get(sort_by, 0) or 0),
                reverse=True
            )
        return results