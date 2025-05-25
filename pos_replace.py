import random
import pickle
import spacy
from collections import defaultdict


def get_random_word_same_pos(target_word, pos_dict_path="pos_dictionary5k.pkl"):
    """
    Get a random word with the same POS tag as the target word.

    Args:
        target_word: The word to match POS tag with
        pos_dict_path: Path to the pickled POS dictionary

    Returns:
        A random word with the same POS tag, or the original word if none found
    """
    # Load the POS dictionary
    with open(pos_dict_path, "rb") as f:
        pos_dict = pickle.load(f)

    # Load spaCy model to get POS tag of target word
    model_path = "/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0"
    try:
        nlp = spacy.load(model_path)
    except:
        nlp = spacy.load("en_core_web_sm")

    # Process the target word to get its POS tag
    target_doc = nlp(target_word)
    target_pos = target_doc[0].pos_

    # Get list of words with the same POS tag
    same_pos_words = pos_dict.get(target_pos, [])

    # Remove target word from candidates if present
    if target_word in same_pos_words:
        same_pos_words.remove(target_word)

    # Return a random word or the original if no matches
    if same_pos_words:
        return random.choice(same_pos_words)
    else:
        return target_word  # Return original if no match found


# Example usage:
if __name__ == "__main__":
    target = "run"
    random_same_pos = get_random_word_same_pos(target)
    print(f"Target word: {target}")
    print(f"Random word with same POS: {random_same_pos}")
