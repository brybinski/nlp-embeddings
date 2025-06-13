#!/opt/anconda/envs/magisterka/bin/python

from transformers import BertTokenizer, BertModel
import torch
from typing import Any
import torch
from abc import ABC, abstractmethod

class Model(ABC):
    tokenizer: Any
    model: Any
    model_name: str
    device: Any
    
    """Abstract base class for all embedding models."""
    
    @abstractmethod
    def get_embeddings(self, sentence=None) -> list:
        """Get embeddings for a given sentence."""
        pass
    
    @abstractmethod
    def tokenize(self, sentence) -> list:
        """Tokenize a sentence into tokens."""
        pass
    
    @abstractmethod
    def reconstruct_sentence(self, tokens) -> str:
        """Reconstruct a sentence from tokens."""
        pass