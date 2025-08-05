
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from models.BERT_model import BERT_model


def test_token_processing():
    model = BERT_model()
    sentence = "a gwgbsh I am a losing my sanity"
    tokens = model.tokenize(sentence)
    embeddings = model.get_embeddings(sentence)
    assert len(embeddings) == len(tokens), "Weights and tokens length mismatch"
    joined_tokens = model.reconstruct_sentence(tokens)
    test = model.tokenizer.tokenize(joined_tokens)
    assert len(test) == len(tokens), "Reconstructed tokens length mismatch"
    
    
    
    
    
