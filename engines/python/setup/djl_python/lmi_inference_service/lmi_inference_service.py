from abc import ABC, abstractmethod

from djl_python.inputs import Input
from djl_python.outputs import Output


class LmiInferenceService(ABC):

    @abstractmethod
    def is_initialized(self) -> bool:
        pass

    @abstractmethod
    def initialize(self, properties: dict):
        pass

    @abstractmethod
    def inference(self, inputs: Input) -> Output:
        pass
