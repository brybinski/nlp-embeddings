import spacy
import pickle
from collections import defaultdict
import os
from tqdm import tqdm

def generate_pos_dictionary(words_file="words.txt", save_pickle=True, pickle_path="pos_dictionary.pkl"):
    """
    Generate a dictionary mapping POS tags to lists of words.
    
    Args:
        words_file: Path to file containing English words (one per line)
        save_pickle: Whether to save the dictionary as a pickle file
        pickle_path: Path where to save the pickle file
        
    Returns:
        Dictionary with POS tags as keys and lists of words as values
    """
    print("Loading spaCy model...")
    model_path = "/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0"
    nlp = spacy.load(model_path)
    nlp.max_length = 1000000
    pos_dict = defaultdict(list)

    print(f"Reading words from {words_file}...")
    with open(words_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    batch_size = 1000
    total_batches = (len(words) + batch_size - 1) // batch_size
    
    print(f"Processing {len(words)} words...")
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "pkl/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for i in tqdm(range(0, len(words[:]), batch_size), total=total_batches):
        batch = words[i:i+batch_size]
        
        # Process each word individually to get accurate POS tags
        for word in batch:
            doc = nlp(word)
            pos = doc[0].pos_  # Get the POS tag of the word
            pos_dict[pos].append(word)
        
        # Save checkpoint every 20 batches
        batch_num = i // batch_size
        if batch_num > 0 and batch_num % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"pos_dict_checkpoint_{batch_num}.pkl")
            print(f"Saving checkpoint at batch {batch_num} to {checkpoint_path}...")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(dict(pos_dict), f)
    
    # Convert defaultdict to regular dict
    result_dict = dict(pos_dict)
    # Print statistics
    print("\nPOS tag distribution:")
    for pos, word_list in result_dict.items():
        print(f"{pos}: {len(word_list)} words")
    
    # Save as pickle if requested
    if save_pickle:
        print(f"Saving dictionary to {pickle_path}...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(result_dict, f)
        print(f"Dictionary saved to {pickle_path}")
    
    return result_dict

def load_pos_dictionary(pickle_path="pos_dictionary.pkl"):
    """
    Load the POS dictionary from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        Dictionary with POS tags as keys and lists of words as values
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Check if dictionary already exists
    if os.path.exists("pos_dictionary.pkl"):
        print("POS dictionary already exists. Loading from file...")
        pos_dict = load_pos_dictionary()
        print("Dictionary loaded.")
    else:
        # Generate dictionary
        pos_dict = generate_pos_dictionary()
    
    # Example usage
    print("\nExample words by POS:")
    for pos, words in pos_dict.items():
        print(f"{pos}: {words[:5]}...")  # Print first 5 words for each POS