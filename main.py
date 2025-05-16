import os
import sys
import transformers
import torch
from model import Model
from LOFO_explainer import LOFO_explainer
from LIME_explainer import LIME_explainer
from SHAP_explainer import SHAP_explainer

def main():
    # TODO: Porównanie wyników XAI z wynikami warstw attention?
    mod = Model("bert-base-uncased")
    sentence = "I like financial river bank"
    embeddings = mod.get_embeddings(sentence)
    explainer = SHAP_explainer(mod)
    results = explainer.explainEmbeddings(sentence, 4)
    print(results)
    
if __name__ == "__main__":
    main()
