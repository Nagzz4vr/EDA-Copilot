from abc import ABC, abstractmethod

class BaseRule(ABC):
    @abstractmethod
    def apply(self, data):
        pass