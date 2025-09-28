main pypi repo is: [brybinski/xaiEmbeddings](https://github.com/brybinski/xaiEmbeddings.git)

# XAI Embeddings

A Python package for explainable AI (XAI) embeddings analysis, using perturbation based explanation methods for transformer-based models. Explanations show how tokens influence position of each other.

## Installation

```bash
pip install xaiembeddings
```

## Requirements for POS-PFI

- **Spacy Model**: path to spacy model e.g.: `en_core_web_trf`
- **Part-of-speech Dictionary**: dictionary with list of words for permutations.

## Available Explainers

- **AttentionExplainer**: Uses attention weights to explain token importance
- **LOOExplainer**: Leave-one-out analysis
- **POSPermutationExplainer**: Part-of-speech based permutation analysis
- **SubsetExplainer**: Subset-based explanation generation

## Models

- **BERTModel**: BERT bert-base-uncased with embedding extraction
- **Model**: Abstract base class for custom model implementations

## Examples:
Sample usage was provided in `examples.ipynb`

## License

This project is licensed under the MIT License.
