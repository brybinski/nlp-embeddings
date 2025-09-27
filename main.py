import os
import sys
import transformers
import torch
from models.BERT_model import BERT_model
from explainers.LOO_explainer import LOO_explainer
from explainers.subset_explainer import subset_explainer
from explainers.POS_permutation_explainer import POS_explainer
from explainers.attention_explainer import BertAttentionExplainer


def main():
    mod = BERT_model("bert-base-uncased")

    pos_pfi = POS_explainer(
        mod,
        pos_dict="/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl",
        spacy="/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0",
        n=100,
    )
    loo = LOO_explainer(mod)
    from distances import euclidean_distance, cosine_distance

    subset = subset_explainer(mod, distance=euclidean_distance, max_subsets=100)
    att = BertAttentionExplainer(mod, aggregation_method="sum")

    docpath = "/home/ryba/Documents/Latex/SEM-DYP-RybinskiBartosz/parts/chapter4"

    # # TODO: TQDM progress bar for every explanation type

    for n, i in enumerate(
        [
            # sentence, token idx
            ("When I was little I caught bass fish", 6),
            ("My beloved music instrument is the bass guitar", 6),
        ]
    ):
        exps = []
        exps.append(loo.explainOne(i[0], i[1]))
        exps.append(pos_pfi.explainOne(i[0], i[1]))
        exps.append(subset.explainOne(i[0], i[1]))
        att.explainOne(i[0], i[1]).plot_comparison(
            i[1],
            *exps,
            show=True,
            save_path=os.path.join(docpath, f"comparison_explanation{n}.png"),
        )

    # # Create explanations with different methods
    # loo_exp = LOO_explainer(mod).explainEmbeddings(sentence)

    # loo_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "loo_self_comparison.html"))
    # shapley_exp = SHAP_explainer(mod).explainEmbeddings(sentence)

    # shapley_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "shap_self_comparison.html"))

    # pos_pfi_exp = pos_pfi.explainEmbeddings(sentence)
    # pos_pfi_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "pos_self_comparison.html"))

    # attention_exp = BertAttentionExplainer(mod).explainEmbeddings(sentence)
    # attention_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "att_self_comparison.html"))
    # # loo_exp.plot_one(6, show=True)


if __name__ == "__main__":
    main()
