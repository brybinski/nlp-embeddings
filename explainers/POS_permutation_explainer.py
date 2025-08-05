import pickle
from explainers.explainer import Explainer
from explainers.explanation import Explanation
from models.model import Model
from distances import cosine_distance,euclidean_distance
import spacy
import random
import copy



class POS_explainer(Explainer):
    """
    Permutation feature importance biorące pod uwagę części mowy
    Zmienia n razy token na inny o tej samej części mowy
    i oblicza odległość do badanego osadzenia.
    W ten sposób można ocenić wpływ danego tokena na osadzenie
    Wykorzystuje słownik części mowy, który jest wczytywany z pliku
    """    
    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance = kwargs.get("distance", cosine_distance)
        path = kwargs.get("pos_dict", None)
        spaCyModel = kwargs.get("spacy", "en_core_web_trf")
        self.pos_tagger = spacy.load(spaCyModel)
        
        # Dictionary with POS tags as keys and lists of words as values
        self.pos_dict:dict = self.load_pos_dict(path) 
        
        # Number of permutations for each token
        self.n = kwargs.get("n", 100)  
        
    def explainEmbeddings(self, sentence, word_range=None, **kwargs) -> Explanation:
        tokens = self.model.tokenizer.tokenize(sentence)
        
        embeddings = self.model.get_embeddings(sentence)

        # POS tag list
        pos_tags = self.tag_tokens(tokens)
        
        # Set word_range if not provided TODO: rename to token_range
        if word_range is None:
            word_range = (0, len(tokens))
        
        # Reconstruct the sentence from the tokens in the specified range
        analyzed_tokens = tokens[word_range[0]:word_range[1]]
        anazlyzed_sentence = self.model.reconstruct_sentence(analyzed_tokens)
        
        # Create an explanation object
        explanation = Explanation("POS permutation", anazlyzed_sentence, analyzed_tokens)

        # Iterate over the specified range of tokens
        for word in range(word_range[0], word_range[1]):
            token = tokens[word]
            
            # Get word scores
            word_scores = self.get_word_score(token, word, pos_tags[word], tokens, embeddings)
            
            # Add scores to the explanation
            for sub_pos, score in enumerate(tokens):
                explanation.add_one_word(token, word, score, sub_pos, word_scores[score])
        
        return explanation
    
    # Load POS dictionary from a file
    def load_pos_dict(self, pos_dict_path):
        with open(pos_dict_path, "rb") as f:
            self.pos_dict = pickle.load(f)
        if not isinstance(self.pos_dict, dict):
            raise ValueError("Loaded POS dictionary is not a dictionary")
        return self.pos_dict
    
    # Tag tokens with their POS tags
    def tag_tokens(self, tokens):
        pos_tags = []
        for token in tokens:
            if token.startswith("##"):
                if  not pos_tags[-1] == ("SUBWORD"):
                    pos_tags[-1] = ("SUBWORD")
                pos_tags.append("SUBWORD")
            else:
                try:
                    doc = self.pos_tagger(token)
                    pos_tags.append(doc[0].pos_)
                except:
                    pos_tags.append("X")
        
        return pos_tags

    # Make a sentence with a change at a specific position
    def make_sentence(self, change, position, tokens):
        copy_tokens = copy.deepcopy(tokens)
        copy_tokens[position] = change
        return self.model.reconstruct_sentence(copy_tokens)
        
    # Calculate the score for a word based on its POS tag
    def get_word_score(self, word, position, pos_tag, tokens, embeddings):
        scores = {}
        for token in tokens:
            scores[token] = 0.0
        
        if pos_tag == 'SUBWORD':
            return scores
        
        if tokens[position] in ',.;':
            return scores
        
        for n in range(self.n):
            replacement = random.choice(self.pos_dict[pos_tag])
            permuted_sentence = self.make_sentence(replacement, position, tokens)
            permuted_embeddings = self.model.get_embeddings(permuted_sentence)
            
            for i, tok in enumerate(tokens):
                distance = self.distance(embeddings[i], permuted_embeddings[i])
                scores[tok] += distance
        
        for key in scores:
            scores[key] /= self.n
            
        return scores
            
        