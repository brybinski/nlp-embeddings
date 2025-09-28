from xai_embeddings.explainers.POS_permutation_explainer import POS_explainer
from xai_embeddings.explainers.LOO_explainer import LOO_explainer
from xai_embeddings.explainers.subset_explainer import subset_explainer
from xai_embeddings.explainers.attention_explainer import BertAttentionExplainer
from xai_embeddings.models.BERT_model import BERT_model
import pandas as pd
import pickle
from tqdm import tqdm
import time
import os
import spacy
import random

random.seed(42)

from xai_embeddings.distances import euclidean_distance


def tag_tokens(tokens):
    try:
        pos_tagger = spacy.load("en_core_web_trf")
    except:
        pos_tagger = spacy.load(
            "/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0"
        )
    pos_tags = []

    for token in tokens:
        if token.startswith("##"):
            if not pos_tags[-1] == ("SUBWORD"):
                pos_tags[-1] = "SUBWORD"
            pos_tags.append("SUBWORD")
        else:
            try:
                doc = pos_tagger(token)
                pos_tags.append(doc[0].pos_)
            except:
                pos_tags.append("X")

    return pos_tags


def get_validation_df(explainers, samples, output_file, mod):

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    header_written = False

    for sentence_idx, sentence in enumerate(tqdm(samples, desc="Processing sentences")):
        explanations = {}
        explainer_times = {}
        token_count = len(mod.tokenizer.tokenize(sentence))

        random_token_pos = random.randint(0, token_count - 1)

        for explainer_name, explainer in tqdm(
            explainers.items(), desc="Processing explainers"
        ):
            tqdm.write(f"Processing explainer: {explainer_name}")
            start_time = time.time()
            expl = explainer.explainOne(sentence, random_token_pos)
            explanations[explainer_name] = expl.normalize()
            end_time = time.time()
            explainer_times[explainer_name] = end_time - start_time

        sentence_records = []

        first_explainer = next(iter(explanations.values()))
        tokens = first_explainer.tokens
        pos_tags = tag_tokens(tokens)
        pos_tag_dict = {pos: tag for pos, tag in enumerate(pos_tags)}

        for explainer_name, explanation in explanations.items():
            for token1_pos, token1_data in explanation.scores.items():
                token1 = token1_data["token"]
                token1_pos_tag = pos_tag_dict.get(token1_pos, "X")

                for token2_pos, token2_data in token1_data["intp"].items():
                    token2 = token2_data["token"]
                    token2_pos_tag = pos_tag_dict.get(token2_pos, "X")
                    score = token2_data["score"]

                    record = {
                        "sentence_idx": sentence_idx,
                        "token1": token1,
                        "token2": token2,
                        "token1_pos": token1_pos,
                        "token2_pos": token2_pos,
                        "token1_pos_tag": token1_pos_tag,
                        "token2_pos_tag": token2_pos_tag,
                        "sentence_length": len(tokens),
                    }

                    for exp_name in explainers.keys():
                        record[exp_name] = None
                        record[f"{exp_name}_time"] = None

                    record[explainer_name] = score
                    record[f"{explainer_name}_time"] = explainer_times[explainer_name]

                    for other_exp_name, other_exp in explanations.items():
                        if other_exp_name == explainer_name:
                            continue

                        if token1_pos in other_exp.scores:
                            token1_data_other = other_exp.scores[token1_pos]
                            if (
                                "intp" in token1_data_other
                                and token2_pos in token1_data_other["intp"]
                            ):
                                record[other_exp_name] = token1_data_other["intp"][
                                    token2_pos
                                ]["score"]
                                record[f"{other_exp_name}_time"] = explainer_times[
                                    other_exp_name
                                ]

                    sentence_records.append(record)

        if sentence_records:
            sentence_df = pd.DataFrame(sentence_records)

            if not header_written:
                sentence_df.to_csv(output_file, index=False, mode="w")
                header_written = True
            else:
                sentence_df.to_csv(output_file, index=False, mode="a", header=False)

            print(f"Saved {len(sentence_records)} records for sentence {sentence_idx}")

        del explanations

    final_df = pd.read_csv(output_file)
    return final_df


if __name__ == "__main__":

    mod = BERT_model("bert-base-uncased")
    explainers = {}
    loo = LOO_explainer(mod)
    att = BertAttentionExplainer(mod, aggregation_method="sum", silent=True)
    for n in range(1, 11, 1):
        shap = subset_explainer(mod, max_subsets=n, silent=True)
        explainers[f"cossubset{n}"] = shap
    for n in range(12, 21, 2):
        shap = subset_explainer(mod, max_subsets=n, silent=True)
        explainers[f"cossubset{n}"] = shap

    for n in range(1, 20, 2):
        pos_pfi = POS_explainer(
            mod,
            pos_dict="/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl",
            spacy="/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0",
            n=n,
            distance=euclidean_distance,
            silent=True,
        )
        explainers[f"l2pospfi{n}"] = pos_pfi

    # l4_loo = LOO_explainer(mod, distance=euclidean_distance)
    # l4_shap = SHAP_explainer(mod, distance=euclidean_distance, max_subsets=20, silent=True)
    # l4_pos_pfi = POS_explainer(
    #     mod,
    #     pos_dict="/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl",
    #     spacy="/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0",
    #     n=80,
    #     distance=euclidean_distance,
    # )

    # explainers["l4_pos_pfi80"] = l4_pos_pfi
    # explainers["l4_loo"] = l4_loo
    # explainers["l4_shap20"] = l4_shap
    # explainers["loo"] = loo
    # explainers["att"] = att

    sample_file = "samples/sample1000.pkl"
    output_file = "dfs/convergencefix.csv"
    with open(sample_file, "rb") as f:
        samples = pickle.load(f)

    explainers_validation = get_validation_df(
        explainers, samples[:100], output_file, mod
    )
    print(explainers_validation.head())
