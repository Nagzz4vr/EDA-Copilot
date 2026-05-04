import hashlib

class CacheManager:
    def __init__(self, storage_provider, current_rule_version: str):
        self.storage=storage_provider
        self.rule_version=current_rule_version
        
    def lookup(self,package:dict):
        metadata=package["metadata"]
        uuid = metadata["state_uuid"]
        fingerprint = metadata["fingerprint"]

        exact_entry = self.storage.find_one({"state_uuid": uuid})
        if exact_entry:
            if self._is_still_valid(exact_entry):
                return "EXACT_HIT", exact_entry["dag"]
            else:
                pass

        candidates = self.storage.find_many(
            {"fingerprint": fingerprint}, 
            sort_by="validation_score"
        )

        if candidates:
            return "PARTIAL_HIT", candidates[0]["dag"]
        
        return "MISS", None

    
