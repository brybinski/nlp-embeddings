import random
import re
import pickle

def word_count(text):
    return len(re.findall(r'\b\w+\b', text))
    
def save_word_count_map(input_file, output_file):
    word_count_map = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            word_count_map[line_number] = word_count(line.strip())
    
    with open(output_file, "wb") as f:
        pickle.dump(word_count_map, f)
        
# save_word_count_map("bookcorpus.txt", "word_count_map.pkl")
    
def load_samples(file_path, n, min_l, max_l, word_count_map_path):
    with open(word_count_map_path, "rb") as f:
        word_count_map = pickle.load(f)

    samples = []
    summary_word_count = 0 
    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line_number in word_count_map:
                word_count_value = word_count_map[line_number]
                if min_l <= word_count_value <= max_l:
                    samples.append(line.strip())
                    summary_word_count += word_count_value
                    
    samples = random.sample(samples, min(n, len(samples)))

    return samples


# samples_debug = load_samples("bookcorpus.txt", n=5, min_l=4, max_l=10, word_count_map_path="word_count_map.pkl")
# with open("samples_debug.pkl", "wb") as f:
#     pickle.dump(samples_debug, f)
    
samples = load_samples("bookcorpus.txt", n=100, min_l=4, max_l=10, word_count_map_path="word_count_map.pkl")

samples_report = ""
for i in range(10, len(samples)+1, 10):
    with open(f"samples/sample{i}.pkl", "wb") as f:
        pickle.dump(samples[:i], f)
        print(f"Saved {i} samples with total word count {sum(word_count(s) for s in samples[:i])}") 
        samples_report += f"Sample {i} total words {sum(word_count(s) for s in samples[:i])}\n"
    
with open(f"samples/report.txt", "w", encoding="utf-8") as f:
    f.write(samples_report)
