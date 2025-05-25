from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Union
from model import Model

class Explainer(ABC):
    model: Model
    
    @abstractmethod
    def __init__(self, model: Model, **kwargs):
        pass
    
    @abstractmethod
    def explainEmbeddings(self, sentence, **kwargs) -> dict:
            pass
    
    # @abstractmethod
    # def compare(self, sentence1, sentence2, **kwargs) -> list:
    #     pass
    