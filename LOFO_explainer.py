from explainer import Explainer
from model import Model
import torch
import copy
from distances import euclidean_distance


# Leave one feature out (LOFO) explainer
# Usuwa jeden feature (token) na raz i oblicza odległość do badanego
# osadzenia. porównując odległości można ocenić wpływ danego tokena na
# osadzenie



class LOFO_explainer(Explainer):
    model: Model

    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance = kwargs.get("distance", euclidean_distance)

    def explainEmbeddings(self, sentence, word_range=None, **kwargs) -> dict:
        tokens = self.model.tokenizer.tokenize(sentence)

        # Check if the number of embeddings matches the number of tokens
        embeddings = self.model.get_embeddings(sentence)
        assert len(embeddings) == len(tokens), "Weights and tokens length mismatch"
        joined_tokens = " ".join(tokens)
        test = self.model.tokenizer.tokenize(joined_tokens)
        assert len(test) == len(tokens), "Tokenization failed"

        dist = {}

        if word_range is None:
            word_range = (0, len(tokens))

        for word in range(word_range[0], word_range[1]):
            dist[word] = [tokens[word]]
            score = [0.0] * len(tokens)
            for num, i in enumerate(tokens):
                if num == word:
                    continue
                modified_tokens = copy.deepcopy(tokens)
                modified_tokens.pop(num)
                new_word_idx = word if word < num else word - 1

                modified_sentence = " ".join(modified_tokens)
                modified_embeddings = self.model.get_embeddings(modified_sentence)
                distance = self.distance(
                    embeddings[word], modified_embeddings[new_word_idx]
                )
                score[num] = distance

            dist[word] = [
                tokens[word],
                {
                    "shapley_values": score,  # TODO: zmienić na score w obu explainerach
                    "explained_token": tokens[word],
                },
            ]

        return dist
