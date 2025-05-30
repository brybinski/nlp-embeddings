import pickle
from explainer import Explainer
from model import Model
from distances import cosine_distance, euclidean_distance
import spacy
import random
import copy


class POS_explainer(Explainer):    
    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance = kwargs.get("distance", cosine_distance)
        path = kwargs.get("pos_dict", None)
        spaCyModel = kwargs.get("spacy", "en_core_web_trf")
        self.pos_tagger = spacy.load(spaCyModel)

        self.pos_dict:dict = self.load_pos_dict(path) # dictionary with POS tags as keys and lists of words as values
        self.n = kwargs.get("n", 100)  # number of permutations for each token
        
    def explainEmbeddings(self, sentence, word_range=None, **kwargs) -> dict:
        tokens = self.model.tokenizer.tokenize(sentence)
        
        # Check if the number of embeddings matches the number of tokens
        embeddings = self.model.get_embeddings(sentence)
        
        
        assert len(embeddings) == len(tokens), "Weights and tokens length mismatch"

        joined_tokens = self.model.reconstruct_sentence(tokens)
        test = self.model.tokenizer.tokenize(joined_tokens)
        assert len(test) == len(tokens), "Tokenization failed"


        # POS tag list
        pos_tags = self.tag_tokens(tokens)
        
        if word_range is None:
            word_range = (0, len(tokens))
            
        score = {}
        for word in range(word_range[0], word_range[1]):
            score[word] = [tokens[word]]
            score[word].append(self.get_word_score(
                tokens[word],
                word,
                pos_tags[word],
                tokens,
                embeddings))
        
        return score

    def load_pos_dict(self, pos_dict_path):
        with open(pos_dict_path, "rb") as f:
            self.pos_dict = pickle.load(f)
        if not isinstance(self.pos_dict, dict):
            raise ValueError("Loaded POS dictionary is not a dictionary")
        return self.pos_dict
    
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

    def make_sentence(self, change, position, tokens):
        copy_tokens = copy.deepcopy(tokens)
        copy_tokens[position] = change
        return self.model.reconstruct_sentence(copy_tokens)
        
        
    def get_word_score(self, word, position, pos_tag, tokens, embeddings):
        scores = {}
        for token in tokens:
            scores[token] = 0.0
        
        if pos_tag == 'SUBWORD':
            # TODO: handle subwords
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
            
        