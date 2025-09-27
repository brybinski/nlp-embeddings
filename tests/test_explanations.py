import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from models.BERT_model import BERT_model
from explainers.LOO_explainer import LOO_explainer
from explainers.subset_explainer import subset_explainer
from explainers.POS_permutation_explainer import POS_explainer
from explainers.attention_explainer import BertAttentionExplainer
from explainers.explanation import Explanation


class TestExplainers:
    model = BERT_model()
    sentence = "I like fish"

    mod = BERT_model("bert-base-uncased")
    pos_pfi = POS_explainer(
        mod,
        pos_dict="/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl",
        spacy="/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0",
        n=100,
    )
    loo = LOO_explainer(mod)
    shap = subset_explainer(mod)
    att = BertAttentionExplainer(mod, aggregation_method="sum")

    loo_expl = loo.explainEmbeddings(sentence)
    shap_expl = shap.explainEmbeddings(sentence)
    pos_expl = pos_pfi.explainEmbeddings(sentence)
    attention_expl = att.explainEmbeddings(sentence)

    explainers = [pos_pfi, loo, shap, att]
    explanations = [pos_expl, loo_expl, shap_expl, attention_expl]

    def test_explanation_instance(self):
        for explanation in self.explanations:
            assert isinstance(
                explanation, Explanation
            ), "Explanation should be an instance of Explanation class"

    def test_explanation_length(self):
        print(len(self.loo_expl.scores))
        for explanation in self.explanations:
            assert len(explanation.scores) > 0, "Explanation should not be empty"


if __name__ == "__main__":
    pytest.main([__file__])
