from explainer import Explainer
from model import Model
from distances import euclidean_distance
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch

# lime text explainer posiada funkcje do obliczania odległości
# dla modeli klasyfikacyjnych ale nie pasuje idealnie do mojego zastosowania
# TODO: muszę przemyśleć jak zaimplementować funkcję predict_proba


class LIME_explainer(Explainer):
    model: Model

    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.distance = kwargs.get("distance", euclidean_distance)
        self.lime_explainer = LimeTextExplainer()

    def predict_proba(self, list_of_texts):
        raise NotImplementedError(
            "LIME requires a predict_proba method to be implemented."
        )

    def explainEmbeddings(self, sentence, word_idx, **kwargs) -> dict:
        raise NotImplementedError(
            "LIME requires a explainEmbeddings method to be implemented."
        )
