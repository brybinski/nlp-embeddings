import spacy
import pickle
from collections import defaultdict
import os
from tqdm import tqdm

# ze 2-3h to przechodziło na 467 tys. słów
def generate_pos_dictionary(words_file="words.txt", save_pickle=True, pickle_path="pos_dictionary.pkl"):
    model_path = "/home/ryba/Documents/Code/snek/magisterka/en_core_web_trf-3.8.0/en_core_web_trf/en_core_web_trf-3.8.0"
    nlp = spacy.load(model_path)
    nlp.max_length = 1000000
    pos_dict = defaultdict(list)

    with open(words_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    batch_size = 1000
    total_batches = (len(words) + batch_size - 1) // batch_size
    
    print(f"Processing {len(words)} words...")
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "pkl/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Check for existing checkpoints to resume processing
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) 
                             if f.startswith("pos_dict_checkpoint_") and f.endswith(".pkl")])
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        batch_num = int(checkpoint_files[-1].split("_")[-1].split(".")[0])
        
        print(f"Found checkpoint from batch {batch_num}, loading...")
        with open(latest_checkpoint, 'rb') as f:
            loaded_dict = pickle.load(f)
            pos_dict.update(loaded_dict)
        
        print(f"Loaded {sum(len(words) for words in loaded_dict.values())} words from checkpoint.")
    else:
        print("No checkpoints found, starting from scratch.")
        batch_num = 0

    start_idx = (batch_num + 1) * batch_size if checkpoint_files else 0
    remaining_batches = (len(words) - start_idx + batch_size - 1) // batch_size
    
    for i in tqdm(range(start_idx, len(words), batch_size), total=remaining_batches):
        batch = words[i:i+batch_size]
        
        for word in batch:
            doc = nlp(word)
            pos = doc[0].pos_ 
            pos_dict[pos].append(word)
        
        batch_num = i // batch_size
        if batch_num > 0 and batch_num % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"pos_dict_checkpoint_{batch_num}.pkl")
            print(f"Saving checkpoint at batch {batch_num} to {checkpoint_path}...")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(dict(pos_dict), f)
    
    result_dict = dict(pos_dict)
    print("\nPOS tag distribution:")
    for pos, word_list in result_dict.items():
        print(f"{pos}: {len(word_list)} words")
    
    if save_pickle:
        print(f"Saving dictionary to {pickle_path}...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(result_dict, f)
        print(f"Dictionary saved to {pickle_path}")
    
    return result_dict

def load_pos_dictionary(pickle_path="pos_dictionary.pkl"):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    if os.path.exists("pos_dictionary.pkl"):
        print("POS dictionary already exists. Loading from file...")
        pos_dict = load_pos_dictionary()
        print("Dictionary loaded.")
    else:
        pos_dict = generate_pos_dictionary()
    
    print("\nExample words by POS:")
    for pos, words in pos_dict.items():
        print(f"{pos}: {words[:5]}...")