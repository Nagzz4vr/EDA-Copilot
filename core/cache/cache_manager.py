import hashlib

class CacheManager:
    def __init__(self, storage_provider, current_rule_version: str):
        self.storage = storage_provider
        self.rule_version = current_rule_version

    def lookup(self, package: dict):
        metadata = package.get("metadata", {})
    
        uuid = metadata.get("state_uuid")
        fingerprint = metadata.get("fingerprint")
    
        # --------------------------------------------------
        # EXACT HIT
        # --------------------------------------------------
    
        exact_entry = self.storage.find_one({
            "state_uuid": uuid
        })
    
        if exact_entry:
            dag = exact_entry.get("dag")
    
            if dag is not None:
                return "EXACT_HIT", dag
    
        # --------------------------------------------------
        # PARTIAL HIT
        # --------------------------------------------------
    
        candidates = self.storage.find_many(
            {"fingerprint": fingerprint},
            sort_by="validation_score"
        )
    
        valid_candidates = [
            c for c in candidates
            if c.get("dag") is not None
        ]
    
        if valid_candidates:
            return "PARTIAL_HIT", valid_candidates[0]["dag"]
    
        # --------------------------------------------------
        # MISS
        # --------------------------------------------------
    
        return "MISS", None
    