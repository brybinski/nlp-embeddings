from explainers.explainer import Explainer
from models.model import Model
import torch
import itertools
import copy
from distances import euclidean_distance, cosine_distance
import numpy as np
from itertools import combinations
import random
import sys
import tqdm
from explainers.explanation import Explanation
from typing import Callable


class subset_explainer(Explainer):
    """
    Tworzy quasi-shapley values dla tokenów w zdaniu
    Wykorzystuje zmodyfikowny algorytm Shapley do obliczenia wartości wpływu każdego
    tokenu na osadzenie. Podejście na bazie subsetów
    """

    model: Model
    silent: bool
    max_subsets: int

    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance: Callable = kwargs.get("distance", cosine_distance)
        self.max_subsets: int = kwargs.get("max_subsets", 100)
        self.silent: bool = kwargs.get("silent", False)

    def explainEmbeddings(self, sentence, word_idx=None, **kwargs) -> Explanation:
        max_subsets: int = self.max_subsets
        tokens: list = self.model.tokenize(sentence)
        original_embeddings = self.model.get_embeddings(sentence)

        # Create explanation object
        explanation = Explanation(f"subset{self.max_subsets}", sentence, tokens)

        # If word_idx is specified, only explain that specific token
        if word_idx is not None:
            if not self.silent:
                print(f'Explaining token: "{tokens[word_idx]}"')

            # Calculate Shapley values for the target token
            shap_vals = self.shap_values(
                self.get_score,
                range(0, len(tokens)),
                max_subsets=max_subsets,
                target=word_idx,
                token_list=tokens,
                distance=self.distance,
                original_embeddings=original_embeddings,
            )

            for i, val in enumerate(shap_vals):
                explanation.add_one_word(
                    tokens[word_idx],  # Main token
                    word_idx,  # Main position
                    tokens[i],  # Sub token
                    i,  # Sub position
                    val,  # Shapley value
                )
        else:
            # Explain all tokens
            for word in range(len(tokens)):
                if not self.silent:
                    print(f'Token "{tokens[word]}": {word+1}/{len(tokens)}')

                # Calculate Shapley values for this token
                shap_vals = self.shap_values(
                    self.get_score,
                    range(0, len(tokens)),
                    max_subsets=max_subsets,
                    target=word,
                    token_list=tokens,
                    distance=self.distance,
                    original_embeddings=original_embeddings,
                )

                # Add Shapley values to the explanation object
                for i, val in enumerate(shap_vals):
                    explanation.add_one_word(
                        tokens[word],  # Main token
                        word,  # Main position
                        tokens[i],  # Sub token
                        i,  # Sub position
                        abs(val),  # Shapley value
                    )

        return explanation

    def get_score(self, perm, **kwargs):
        token_list: list = kwargs.get("token_list", [])
        target: int = kwargs.get("target", -1)
        distance: Callable = kwargs.get(
            "distance",
            lambda: (_ for _ in ()).throw(
                RuntimeError("No distance function provided")
            ),
        )
        original_embeddings: list = kwargs.get("original_embeddings", [])
        perm = list(perm)

        target_perm_idx = perm.index(target)

        test_tokens = []
        for i in range(len(perm)):
            test_tokens.append(token_list[perm[i]])

        test_sentence = " ".join(test_tokens)

        new_emb = self.model.get_embeddings(test_sentence)
        target_embedding = original_embeddings[target]
        target_perm_embedding = new_emb[target_perm_idx]

        distance_value = distance(target_embedding, target_perm_embedding)
        return distance_value

    def shap_values(
        self, model_foo, input_x, baseline=None, max_subsets=None, **kwargs
    ):
        target: int = kwargs.get("target", -1)
        token_list: list = kwargs.get("token_list", [])

        n = len(input_x)
        if baseline is None:
            baseline = list(range(n))

        shap_vals = [0.0] * n

        all_subsets = gensubsets(input_x, present=target, max_subset_size=max_subsets)

        for subset in (
            all_subsets
            if self.silent
            else tqdm.tqdm(
                all_subsets, desc=f"{token_list[target]} SHAP", unit="subset"
            )
        ):

            if len(subset) == 0:
                continue
            f_S = model_foo(subset, **kwargs)
            for i in range(n):
                if i not in subset:
                    S_with_i = subset.union({i})
                    f_Si = model_foo(S_with_i, **kwargs)
                    shap_vals[i] += f_Si - f_S

        # TODO: Przemyśleć jak sprawdzić co działa pozytywnie a co negatywnie
        # Na dane osadzenie
        shap_vals = [s / len(all_subsets) for s in shap_vals]
        return shap_vals

    def explainOne(self, sentence, position, **kwargs) -> Explanation:
        return self.explainEmbeddings(sentence, word_idx=position, **kwargs)


def gensubsets(s, present=None, max_subset_size=None):
    if max_subset_size is None:
        max_subset_size = sys.maxsize

    if present is None:
        subsets = []
        for i in range(len(s) + 1):
            for subset in combinations(s, i):
                subsets.append(set(subset))

    else:
        present_element = s[present]
        other_elements = [s[i] for i in range(len(s)) if i != present]

        subsets = []
        for i in range(len(other_elements) + 1):
            for subset in combinations(other_elements, i):
                new_subset = set(subset)
                new_subset.add(present_element)
                subsets.append(new_subset)

    if len(subsets) >= max_subset_size:
        return random.sample(subsets, max_subset_size)

    return subsets
