import pickle
import pandas as pd
import random
import copy
import os
from tqdm import tqdm
from explainers.POS_permutation_explainer import POS_explainer
from explainers.subset_explainer import subset_explainer
from models.BERT_model import BERT_model
from explainers.LOO_explainer import LOO_explainer
from explainers.attention_explainer import BertAttentionExplainer
import spacy
import random


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


def remove_tokens_from_sentence(sentence, remove, model):
    tokens = model.tokenize(sentence)
    sorted_positions = sorted(remove, reverse=True)
    modified_tokens = tokens[:]
    for pos in sorted_positions:
        if 0 <= pos < len(modified_tokens):
            modified_tokens.pop(pos)
    modified_sentence = model.reconstruct_sentence(modified_tokens)
    return modified_sentence, modified_tokens


def insdel(samples, explainers, k=3, output_file="prog_ins_del.csv"):

    results = []
    write_header = not os.path.exists(output_file)
    for sentence_idx, sentence in enumerate(tqdm(samples, desc="Processing sentences")):
        try:
            print(f"\nProcessing sentence {sentence_idx}: {sentence}")

            for explainer_name, explainer in explainers.items():
                try:
                    tokens = explainer.model.tokenize(sentence)
                    pos_tags = tag_tokens(tokens)

                    random_token_pos = random.choice(range(1, len(tokens) - 1))
                    random_token = tokens[random_token_pos]
                    random_token_pos_tag = pos_tags[random_token_pos]

                    print(
                        f"Selected token: {random_token} at position {random_token_pos}, tag: {random_token_pos_tag}"
                    )

                    original_explanation = explainer.explainOne(
                        sentence, random_token_pos
                    ).normalize()

                    if random_token_pos not in original_explanation.scores:
                        print(
                            f"No explanation found for token at position {random_token_pos}"
                        )
                        continue

                    influences = original_explanation.scores[random_token_pos]["intp"]

                    if len(influences) == 0:
                        print(f"No influences found for token {random_token}")
                        continue

                    most_influential_pos = random.randint(0, len(tokens) - 1)
                    most_influential_token = influences[most_influential_pos]["token"]
                    original_score = influences[most_influential_pos]["score"]

                    print(
                        f"Most influential token: {most_influential_token} (pos {most_influential_pos}) with score {original_score}"
                    )

                    other_influences = {
                        pos: data
                        for pos, data in influences.items()
                        if pos != random_token_pos and pos != most_influential_pos
                    }

                    if len(other_influences) < k:
                        print(
                            f"Not enough other influential tokens (found {len(other_influences)}, need {k})"
                        )
                        k_actual = len(other_influences)
                    else:
                        k_actual = k

                    top_k_positions = sorted(
                        other_influences.keys(),
                        key=lambda pos: abs(other_influences[pos]["score"]),
                        reverse=True,
                    )[:k_actual]

                    print(
                        f"Top {k_actual} influential tokens to remove: {[tokens[pos] for pos in top_k_positions]}"
                    )

                    result = {
                        "sentence_idx": sentence_idx,
                        "most_influential_token": most_influential_token,
                        "pos_tag": random_token_pos_tag,
                        "explainer": explainer_name,
                        "orig": original_score,
                    }

                    removed_positions = []
                    current_sentence = sentence

                    for i in range(k_actual):
                        removed_positions.append(top_k_positions[i])
                        modified_sentence, modified_tokens = (
                            remove_tokens_from_sentence(
                                sentence, removed_positions, explainer.model
                            )
                        )

                        if len(modified_tokens) <= 1:
                            result[f"del_{i+1}"] = None
                        else:
                            try:

                                modified_pos_mapping = {}
                                original_pos = 0
                                modified_pos = 0
                                original_tokens = tokens

                                for orig_pos in range(len(original_tokens)):
                                    if orig_pos not in removed_positions:
                                        modified_pos_mapping[orig_pos] = modified_pos
                                        modified_pos += 1

                                if most_influential_pos in modified_pos_mapping:
                                    new_most_influential_pos = modified_pos_mapping[
                                        most_influential_pos
                                    ]

                                    if random_token_pos in modified_pos_mapping:
                                        new_target_pos = modified_pos_mapping[
                                            random_token_pos
                                        ]

                                        modified_explanation = explainer.explainOne(
                                            modified_sentence, new_target_pos
                                        ).normalize()

                                        if (
                                            new_target_pos
                                            in modified_explanation.scores
                                            and new_most_influential_pos
                                            in modified_explanation.scores[
                                                new_target_pos
                                            ]["intp"]
                                        ):
                                            modified_score = (
                                                modified_explanation.scores[
                                                    new_target_pos
                                                ]["intp"][new_most_influential_pos][
                                                    "score"
                                                ]
                                            )
                                            result[f"del_{i+1}"] = modified_score
                                        else:
                                            result[f"del_{i+1}"] = None
                                    else:
                                        result[f"del_{i+1}"] = None
                                else:
                                    result[f"del_{i+1}"] = None
                            except Exception as e:
                                print(f"Error in deletion step {i+1}: {e}")
                                result[f"del_{i+1}"] = None

                    for i in range(k_actual):
                        current_removed = removed_positions[: k_actual - i]

                        if len(current_removed) == 0:
                            current_sentence = sentence
                            current_tokens = tokens
                        else:
                            modified_sentence, current_tokens = (
                                remove_tokens_from_sentence(
                                    sentence, current_removed, explainer.model
                                )
                            )

                        if len(current_tokens) <= 1:
                            result[f"ins_{i+1}"] = None
                        else:
                            try:
                                modified_pos_mapping = {}
                                modified_pos = 0

                                for orig_pos in range(len(tokens)):
                                    if orig_pos not in current_removed:
                                        modified_pos_mapping[orig_pos] = modified_pos
                                        modified_pos += 1

                                if (
                                    most_influential_pos in modified_pos_mapping
                                    and random_token_pos in modified_pos_mapping
                                ):
                                    new_most_influential_pos = modified_pos_mapping[
                                        most_influential_pos
                                    ]
                                    new_target_pos = modified_pos_mapping[
                                        random_token_pos
                                    ]

                                    if len(current_removed) == 0:
                                        current_sentence = sentence
                                    else:
                                        current_sentence, _ = (
                                            remove_tokens_from_sentence(
                                                sentence,
                                                current_removed,
                                                explainer.model,
                                            )
                                        )

                                    modified_explanation = explainer.explainOne(
                                        current_sentence, new_target_pos
                                    ).normalize()

                                    if (
                                        new_target_pos in modified_explanation.scores
                                        and new_most_influential_pos
                                        in modified_explanation.scores[new_target_pos][
                                            "intp"
                                        ]
                                    ):
                                        modified_score = modified_explanation.scores[
                                            new_target_pos
                                        ]["intp"][new_most_influential_pos]["score"]
                                        result[f"ins_{i+1}"] = modified_score
                                    else:
                                        result[f"ins_{i+1}"] = None
                                else:
                                    result[f"ins_{i+1}"] = None
                            except Exception as e:
                                print(f"Error in insertion step {i+1}: {e}")
                                result[f"ins_{i+1}"] = None

                    # Fill missing columns
                    for i in range(k_actual + 1, k + 1):
                        result[f"del_{i}"] = None
                    for i in range(k_actual, k):
                        result[f"ins_{i}"] = None

                    results.append(result)

                    result_df = pd.DataFrame([result])
                    if write_header:
                        result_df.to_csv(output_file, index=False, mode="w")
                        write_header = False
                    else:
                        result_df.to_csv(
                            output_file, index=False, mode="a", header=False
                        )

                    print(
                        f"Saved result for sentence {sentence_idx}, explainer {explainer_name}"
                    )

                except Exception as e:
                    print(f"Error processing explainer {explainer_name}: {e}")
                    continue

        except Exception as e:
            print(f"Error processing sentence {sentence_idx}: {e}")
            continue

    try:
        df = pd.read_csv(output_file)
        print(f"\nCompleted analysis. Total records saved: {len(df)}")
    except:
        df = pd.DataFrame(results)
        print(f"\nCompleted analysis. No valid results were saved.")

    return df


def main():
    with open("samples/sample800.pkl", "rb") as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")
    model = BERT_model("bert-base-uncased")
    from distances import euclidean_distance

    explainers = {}
    explainers["attention"] = BertAttentionExplainer(model, silent=True)
    explainers["loo"] = LOO_explainer(model, silent=True)
    explainers["l4_loo"] = LOO_explainer(
        model, silent=True, distance=euclidean_distance
    )

    for i in [20, 40, 60]:
        pos_pfi = POS_explainer(
            model,
            pos_dict="/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl",
            spacy="/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0",
            n=40,
            silent=True,
            distance=euclidean_distance,
        )
        explainers[f"l2_pospfi{i}"] = pos_pfi

    for i in [20, 40, 60]:
        pos_pfi = POS_explainer(
            model,
            pos_dict="/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl",
            spacy="/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0",
            n=40,
            silent=True,
        )
        explainers[f"pospfi{i}"] = pos_pfi

    for shap_n in [20, 40, 60]:
        shap_explainer = subset_explainer(model, max_subsets=shap_n, silent=True)
        explainers[f"subset{shap_n}"] = shap_explainer

    for shap_n in [20, 40, 60]:
        shap_explainer = subset_explainer(model, max_subsets=shap_n, silent=True)
        explainers[f"l4_subset{shap_n}"] = shap_explainer

    k = 3
    output_file = "dfs/insdel_random4.csv"
    for i in range(len(samples)):
        samples[i] = samples[i].replace("``", "").strip()
    try:
        results_df = insdel(samples, explainers, k, output_file)
        print(f"\nAnalysis completed. Generated {len(results_df)} records.")
        print("\nFirst few records:")
        print(results_df.head())
    except Exception as e:
        print(f"Error in main analysis: {e}")
        raise


if __name__ == "__main__":
    main()
