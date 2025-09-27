from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Union
from models.model import Model
from explainers.explanation import Explanation


class Explainer(ABC):
    model: Model
    
    @abstractmethod
    def __init__(self, model: Model, **kwargs):
        pass
    
    @abstractmethod
    def explainEmbeddings(self, sentence, **kwargs) -> Explanation:
            pass
        
    @abstractmethod
    def explainOne(self, sentence, position, **kwargs) -> Explanation:
            pass