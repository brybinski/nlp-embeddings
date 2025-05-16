from explainer import Explainer
from model import Model
import torch
import itertools
import copy
from distances import euclidean_distance
import numpy as np
from itertools import combinations
import random
import sys
import tqdm


class SHAP_explainer(Explainer):
    model: Model

    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance = kwargs.get("distance", euclidean_distance)

    
    def explainEmbeddings(self, sentence, word_idx=None, max_subsets=None, **kwargs) -> dict:
        tokens = self.model.tokenizer.tokenize(sentence)
        
        dist = {}
        original_embeddings = self.model.get_embeddings(sentence)
        
        
        # TODO: Zunifikowana klasa wyjaśnienia pod wizualizację?
        if word_idx is not None:
            word = word_idx
            dist[word] = [tokens[word]]
            shap_vals = self.shap_values(self.get_score, range(0, len(tokens)), max_perms=max_subsets, target=word_idx, token_list=tokens, distance=self.distance, original_embeddings=original_embeddings)
            dist[word].append(
                {
                    "shapley_values": shap_vals,

                    "explained_token": tokens[word],                
                }
            )
            return dist
    
        it = 1
        for word in range(len(tokens)):
            print(f"Token \"{tokens[word]}\": {it}/{len(tokens)}")
            
            dist[word] = [tokens[word]]
            shap_vals = self.shap_values(self.get_score, range(0, len(tokens)), max_subsets=max_subsets, target=word, token_list=tokens, distance=self.distance, original_embeddings=original_embeddings)
            dist[word].append(
                {
                    "shapley_values": shap_vals,

                    "explained_token": tokens[word],                
                }
            )
            it += 1
            
        return dist

    def get_score(self, perm, **kwargs):
        token_list = kwargs.get("token_list")
        target = kwargs.get("target")
        distance = kwargs.get("distance")
        original_embeddings = kwargs.get("original_embeddings")
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
        
    def shap_values(self, model_foo, input_x, baseline=None, max_subsets=None, **kwargs):
            target = kwargs.get("target")
            token_list = kwargs.get("token_list")

            n = len(input_x)
            if baseline is None:
                baseline = list(range(n))
            
            shap_vals = [0.0] * n

            all_subsets = gensubsets(input_x, present=target, max_subset_size=max_subsets)


            for subset in tqdm.tqdm(all_subsets, desc=f"{token_list[target]} SHAP", unit="subset"):

                if len(subset) == 0:
                    continue
                f_S = model_foo(subset, **kwargs)
                for i in range(n):
                    if i not in subset:
                        S_with_i = subset.union({i})
                        f_Si = model_foo(S_with_i, **kwargs)
                        shap_vals[i] += (f_Si - f_S)

            # TODO: Przemyśleć jak sprawdzić co działa pozytywnie a co negatywnie
            # Na dane osadzenie
            shap_vals = [s / len(all_subsets) for s in shap_vals]
            return shap_vals
        
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