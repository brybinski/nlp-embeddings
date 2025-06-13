from explainers.explainer import Explainer
from models.model import Model
import torch
import copy
from distances import euclidean_distance, cosine_distance
from explainers.explanation import Explanation


# Leave one feature out (LOFO) explainer
# Usuwa jeden feature (token) na raz i oblicza odległość do badanego
# osadzenia. porównując odległości można ocenić wpływ danego tokena na
# osadzenie



class LOO_explainer(Explainer):
    model: Model

    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance = kwargs.get("distance", cosine_distance)

    def explainEmbeddings(self, sentence, word_range=None, **kwargs) -> Explanation:
        tokens = self.model.tokenizer.tokenize(sentence)

        embeddings = self.model.get_embeddings(sentence)
        assert len(embeddings) == len(tokens), "Weights and tokens length mismatch"
        joined_tokens = self.model.reconstruct_sentence(tokens)
        test = self.model.tokenizer.tokenize(joined_tokens)
        assert len(test) == len(tokens), "Tokenization failed"

        explanation = Explanation("LOO", sentence, tokens)

        if word_range is None:
            word_range = (0, len(tokens))
            
        for word in range(word_range[0], word_range[1]):
            target_token = tokens[word]
            
            # Calculate influence scores for each token
            for num, token in enumerate(tokens):
                # Skip self-influence
                if num == word:
                    continue
                    
                # Create a modified version without the current token
                modified_tokens = copy.deepcopy(tokens)
                modified_tokens.pop(num)
                
                # Adjust index if needed after removal
                new_word_idx = word if word < num else word - 1
                
                # Get modified embeddings
                modified_sentence = self.model.reconstruct_sentence(modified_tokens)
                modified_embeddings = self.model.get_embeddings(modified_sentence)
                
                # Calculate distance (impact of removing this token)
                distance = self.distance(
                    embeddings[word], modified_embeddings[new_word_idx]
                )
                
                # Add score to explanation object
                explanation.add_one_word(target_token, word, token, num, distance)

        return explanation
    
    
        # for word in range(word_range[0], word_range[1]):
        #     dist[word] = [tokens[word]]
        #     score = [0.0] * len(tokens)
        #     for num, i in enumerate(tokens):
        #         if num == word:
        #             continue
        #         modified_tokens = copy.deepcopy(tokens)
        #         modified_tokens.pop(num)
        #         new_word_idx = word if word < num else word - 1

        #         modified_sentence = " ".join(modified_tokens)
        #         modified_embeddings = self.model.get_embeddings(modified_sentence)
        #         distance = self.distance(
        #             embeddings[word], modified_embeddings[new_word_idx]
        #         )
        #         score[num] = distance

        #     dist[word] = [
        #         tokens[word],
        #         {
        #             "shapley_values": score,  # TODO: zmienić na score w obu explainerach
        #             "explained_token": tokens[word],
        #         },
        #     ]

        # return dist
