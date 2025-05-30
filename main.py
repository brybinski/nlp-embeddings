import os
import sys
import transformers
import torch
from model import Model
from LOFO_explainer import LOFO_explainer
from LIME_explainer import LIME_explainer
from SHAP_explainer import SHAP_explainer
from POS_permutation_explainer import POS_explainer


def main():
    mod = Model("bert-base-uncased")
    sentence = "I like financial river bank."
    embeddings = mod.get_embeddings(sentence)
    explainer = POS_explainer(
        mod,
        pos_dict="/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl",
        spacy="/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0",
        n=10,
    )
    results = explainer.explainEmbeddings(sentence)
    print(results)


if __name__ == "__main__":
    main()
