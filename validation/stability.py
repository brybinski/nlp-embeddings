import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict

from xai_embeddings.explainers.POS_permutation_explainer import POS_explainer
from xai_embeddings.explainers.LOO_explainer import LOO_explainer
from xai_embeddings.explainers.subset_explainer import subset_explainer
from xai_embeddings.explainers.attention_explainer import BertAttentionExplainer
from xai_embeddings.models.BERT_model import BERT_model


# Mostly LLM generated
# I checked code to make sure it makes sense
# and fixed some minor issues
# but I didn't write it all myself
# It's called stability but it's really about consistency
# because stability would imply small changes to input
# this is repeatability. Stability was inserted badly by LLM
def calculate_stability_across_samples(
    explainers, sentences, token_positions, runs_per_sentence=5
):

    # Store all explanations for each explainer
    all_explanations = {name: [] for name in explainers.keys()}
    # Store run times
    run_times = {name: [] for name in explainers.keys()}

    # Process each sentence
    for i, (sentence, token_pos) in enumerate(zip(sentences, token_positions)):
        print(f"\nProcessing sentence {i+1}/{len(sentences)}")
        print(f"Sentence: {sentence}")
        print(f"Token position: {token_pos}")

        # Run each explainer multiple times on this sentence
        for run in tqdm(range(runs_per_sentence), desc=f"Runs for sentence {i+1}"):
            for explainer_name, explainer in explainers.items():
                start_time = time.time()
                explanation = explainer.explainOne(sentence, token_pos)
                normalized_explanation = explanation.normalize()
                all_explanations[explainer_name].append(normalized_explanation)
                end_time = time.time()

                # Record run time
                run_times[explainer_name].append(end_time - start_time)

    # Calculate stability metrics
    stability_metrics = {}

    for explainer_name, explanations in all_explanations.items():
        # Get all token pairs that were scored across all explanations
        token_pairs = set()
        for expl in explanations:
            for token1_pos, token1_data in expl.scores.items():
                for token2_pos in token1_data["intp"].keys():
                    token_pairs.add((token1_pos, token2_pos))

        # Create a matrix to store scores for each token pair across runs
        pair_scores = {pair: [] for pair in token_pairs}

        # Collect scores for each token pair across all runs
        for expl in explanations:
            for pair in token_pairs:
                token1_pos, token2_pos = pair
                score = 0.0  # Default if pair not found

                if (
                    token1_pos in expl.scores
                    and token2_pos in expl.scores[token1_pos]["intp"]
                ):
                    score = expl.scores[token1_pos]["intp"][token2_pos]["score"]

                pair_scores[pair].append(score)

        # Calculate standard deviation for each token pair
        pair_stds = {
            pair: np.std(scores) for pair, scores in pair_scores.items() if scores
        }

        # Calculate coefficient of variation for each token pair
        pair_cvs = {}
        for pair, scores in pair_scores.items():
            if not scores:
                continue
            mean = np.mean(scores)
            std = np.std(scores)
            if mean != 0:
                pair_cvs[pair] = std / abs(mean)
            else:
                pair_cvs[pair] = 0 if std == 0 else float("inf")

        # Calculate average standard deviation and coefficient of variation
        avg_std = np.mean(list(pair_stds.values())) if pair_stds else 0
        avg_cv = (
            np.mean([cv for cv in pair_cvs.values() if cv != float("inf")])
            if pair_cvs
            else 0
        )

        # Calculate rank consistency
        # Group explanations by sentence
        sentence_groups = [
            explanations[i : i + runs_per_sentence]
            for i in range(0, len(explanations), runs_per_sentence)
        ]

        # Calculate rank correlations for each sentence
        all_rank_correlations = []

        for sentence_explanations in sentence_groups:
            # For each run, rank the token pairs by importance
            ranked_lists = []
            for expl in sentence_explanations:
                run_scores = {}
                for pair in token_pairs:
                    token1_pos, token2_pos = pair
                    if (
                        token1_pos in expl.scores
                        and token2_pos in expl.scores[token1_pos]["intp"]
                    ):
                        run_scores[pair] = expl.scores[token1_pos]["intp"][token2_pos][
                            "score"
                        ]
                    else:
                        run_scores[pair] = 0.0

                ranked_pairs = sorted(
                    run_scores.items(), key=lambda x: x[1], reverse=True
                )
                ranked_lists.append([pair for pair, _ in ranked_pairs])

            # Calculate rank correlation between consecutive runs for this sentence
            for i in range(len(ranked_lists) - 1):
                corr = spearman_rank_correlation(ranked_lists[i], ranked_lists[i + 1])
                all_rank_correlations.append(corr)

        avg_rank_correlation = (
            np.mean(all_rank_correlations) if all_rank_correlations else 0
        )

        # Store metrics
        stability_metrics[explainer_name] = {
            "avg_std": avg_std,
            "avg_cv": avg_cv,
            "rank_stability": avg_rank_correlation,
            "avg_time": np.mean(run_times[explainer_name]),
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(stability_metrics, orient="index")

    return metrics_df, all_explanations


from scipy.stats import spearmanr


def spearman_rank_correlation(list1, list2):
    """Calculate Spearman rank correlation between two ranked lists using scipy."""

    # Create positional mappings
    map1 = {item: idx for idx, item in enumerate(list1)}
    map2 = {item: idx for idx, item in enumerate(list2)}

    # Find common elements
    common = list(set(map1.keys()) & set(map2.keys()))

    if not common:
        return 0

    ranks1 = [map1[item] for item in common]
    ranks2 = [map2[item] for item in common]

    rho, _ = spearmanr(ranks1, ranks2)
    return rho


def visualize_stability(metrics_df, output_file="stability_metrics.png"):
    """Visualize stability metrics for different explainers."""
    plt.figure(figsize=(12, 8))

    # Create a heatmap
    sns.heatmap(metrics_df, annot=True, cmap="viridis", fmt=".4f")

    plt.title("Stability Metrics for Different Explainers")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    # Create bar plots for each metric
    metrics = metrics_df.columns
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics_df.index, y=metrics_df[metric])
        plt.title(f"{metric} for Different Explainers")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{metric}_comparison.png")
        plt.close()


def get_valid_token_position(model, sentence):
    """Get a random valid token position for a sentence"""
    tokens = model.tokenizer.tokenize(sentence)
    if len(tokens) <= 2:  # Need at least 3 tokens (start, end, and one in the middle)
        return None
    # Choose a position that's not the start or end token
    return random.randint(1, len(tokens) - 2)


if __name__ == "__main__":
    # Load model
    mod = BERT_model("bert-base-uncased")
    from xai_embeddings.distances import euclidean_distance

    # Setup explainers
    explainers = {}
    # for n in [5, 10, 21, 30, 61]:
    #     explainers[f"cossubset{n}"] = SHAP_explainer(mod, max_subsets=n, silent=True)

    # Add POS PFI explainers with different sample sizes
    pos_dict_path = "/home/ryba/Documents/Code/snek/magisterka/pos_dictionary.pkl"
    spacy_path = "/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0"

    for n in [2, 8]:
        explainers[f"l2pospfi{n}"] = POS_explainer(
            mod,
            pos_dict=pos_dict_path,
            spacy_model=spacy_path,
            max_subsets=n,
            silent=True,
            distance=euclidean_distance,
        )

    # Load sample sentences
    sample_file = "samples/sample1000.pkl"
    with open(sample_file, "rb") as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")
    for sample in samples:
        sample.replace("``", "")
    num_samples = 3
    runs_per_sentence = 5

    # Take a subset of samples if needed
    if num_samples < len(samples):
        selected_samples = random.sample(samples, num_samples)
    else:
        selected_samples = samples
        num_samples = len(samples)

    # Find a valid token position for each sample
    valid_samples = []
    valid_positions = []

    print("Finding valid token positions...")
    for sentence in tqdm(selected_samples):
        token_pos = get_valid_token_position(mod, sentence)
        if token_pos is not None:
            valid_samples.append(sentence)
            valid_positions.append(token_pos)

    print(f"Using {len(valid_samples)} valid samples out of {num_samples} selected")

    # For each sample, print the selected token
    for i, (sentence, token_pos) in enumerate(zip(valid_samples, valid_positions)):
        tokens = mod.tokenizer.tokenize(sentence)
        print(f"\nSample {i+1}:")
        print(f"Sentence: {sentence}")
        print(f"Selected token position: {token_pos} ('{tokens[token_pos]}')")

    # Calculate stability metrics
    print(f"\nAnalyzing stability across {len(valid_samples)} sentences...")
    metrics_df, all_explanations = calculate_stability_across_samples(
        explainers, valid_samples, valid_positions, runs_per_sentence
    )

    # Print stability metrics
    print("\nStability Metrics:")
    print(metrics_df)

    # Save metrics to CSV
    metrics_df.to_csv("dfs/stability_multisample.csv")

    print(
        "\nStability analysis complete. Results saved to dfs/stability_multisample.csv and visualized in PNG files."
    )
