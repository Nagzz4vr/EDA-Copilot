from typing import Dict, List, Any
import tiktoken
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ContextBlob:
    signals: List[Dict[str, Any]]
    total_count: int
    included: int
    dropped: int
    token_cost: int


class ContextCompressor:
    TOKEN_BUDGET = 2000

    @staticmethod
    def compress(ranked_signals: list, model="gpt-4o-mini") -> ContextBlob:
        blob = []
        tokens_used = 0

        enc = tiktoken.encoding_for_model(model)

        for signal in ranked_signals:
            serialized_dict = signal.to_compact_dict()

            serialized_str = json.dumps(serialized_dict, separators=(",", ":"))

            token_cost = len(enc.encode(serialized_str))

            if tokens_used + token_cost > ContextCompressor.TOKEN_BUDGET:
                break

            blob.append(serialized_dict)
            tokens_used += token_cost

        return ContextBlob(
            signals=blob,
            total_count=len(ranked_signals),
            included=len(blob),
            dropped=len(ranked_signals) - len(blob),
            token_cost=tokens_used
        )