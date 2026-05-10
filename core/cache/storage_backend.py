from typing import Protocol, Dict, Any


class StorageBackend(Protocol):

    def write(self, key: str, value: Dict[str, Any]) -> None:
        ...

    def read(self, key: str):
        ...