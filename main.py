import os
import sys
import transformers
import torch
from models.BERT_model import BERT_model
from explainers.LOO_explainer import LOO_explainer
from explainers.SHAP_explainer import SHAP_explainer
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
    shap = SHAP_explainer(mod)
    att = BertAttentionExplainer(mod, aggregation_method="sum")

    docpath = '/home/ryba/Documents/Latex/SEM-DYP-RybinskiBartosz/parts/chapter4'
    
    # # TODO: TQDM progress bar for every explanation type
    
    # for n, i in enumerate([
    #     # sentence, token idx
    #     # ("When I was little I caught bass fish", 6),
    #     # ("My beloved music instrument is the bass guitar", 6),
    # ]):
    #     if len(i) == 3:
    #         picnum = i[2]
    #     else:
    #         picnum = n
    
    #     loo.explainEmbeddings(i[0]).plot_one(i[1], save_path=os.path.join(docpath, f"loo_explanation{picnum}.png"))
        
    #     shap.explainEmbeddings(i[0], i[1]).plot_one(i[1], save_path=os.path.join(docpath, f"shap_explanation{picnum}.png"))
        
    #     pos_pfi.explainEmbeddings(i[0]).plot_one(i[1], save_path=os.path.join(docpath, f"pos_explanation{picnum}.png"))
        
    #     att.explainEmbeddings(i[0], i[1]).plot_one(i[1], save_path=os.path.join(docpath, f"att_explanation{picnum}.png"))

    sentence = "When I was little I caught bass fish"
    # Create explanations with different methods
    loo_exp = LOO_explainer(mod).explainEmbeddings(sentence)
    
    loo_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "loo_self_comparison.html"))
    shapley_exp = SHAP_explainer(mod).explainEmbeddings(sentence)

    shapley_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "shap_self_comparison.html"))
    
    pos_pfi_exp = pos_pfi.explainEmbeddings(sentence)
    pos_pfi_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "pos_self_comparison.html"))
    
    attention_exp = BertAttentionExplainer(mod).explainEmbeddings(sentence)
    attention_exp.plot_self_comparison(show=True, save_path=os.path.join(docpath, "att_self_comparison.html"))
    # loo_exp.plot_one(6, show=True)
    
if __name__ == "__main__":
    main()
