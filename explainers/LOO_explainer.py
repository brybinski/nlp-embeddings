from explainers.explainer import Explainer
from models.model import Model
import torch
import copy
from distances import euclidean_distance, cosine_distance
from explainers.explanation import Explanation




class LOO_explainer(Explainer):
    """
    Leave one feature out (LOFO) explainer
    Usuwa jeden feature (token) na raz i oblicza odległość do badanego
    osadzenia. porównując odległości można ocenić wpływ danego tokenu na
    osadzenie
    """
    model: Model

    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance = kwargs.get("distance", cosine_distance)
        

    def explainEmbeddings(self, sentence, word_range=None, **kwargs) -> Explanation:
        tokens = self.model.tokenizer.tokenize(sentence)

        # Get embeddings for the original sentence
        embeddings = self.model.get_embeddings(sentence)

        # Create an explanation object
        explanation = Explanation("LOO", sentence, tokens)

        # If word_range is not provided, consider all tokens
        if word_range is None:
            word_range = (0, len(tokens))
        
        # Iterate over the specified range of tokens
        for word in range(word_range[0], min(word_range[1], len(tokens))):
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
                
                # Handle cases where the modified embeddings might be shorter
                # because of subword tokenization
                if new_word_idx >= len(modified_embeddings):
                    continue
                
                # Calculate distance (impact of removing this token)
                distance = self.distance(
                    embeddings[word], modified_embeddings[new_word_idx]
                )
                
                # Add score to explanation object
                explanation.add_one_word(target_token, word, token, num, distance)

        return explanation
    
    def explainOne(self, sentence, position, **kwargs) -> Explanation:
        return self.explainEmbeddings(sentence, word_range=(position, position + 1), **kwargs)