import os
import json

class StateRegistry:
    def __init__(self,cache_dir="cache_storage"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    def find_one(self, query):
        uuid = query.get("state_uuid")
        path = os.path.join(self.cache_dir, f"{uuid}.json")
        if os.path.exists(path):
            with open(path, 'r',encoding="utf-8") as f:
                return json.load(f)
        return None
    def find_many(self, query, sort_by=None):
        fingerprint = query.get("fingerprint")
        results = []
        for filename in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, filename)
        
            if not filename.endswith(".json"):
                continue
            
            with open(path, "r", encoding="utf-8") as f:
                try:
                    entry = json.load(f)
                except Exception:
                    continue
        
        if sort_by:
            results.sort(
            key=lambda x: float(x.get(sort_by, 0) or 0),
            reverse=True
        )
        return results