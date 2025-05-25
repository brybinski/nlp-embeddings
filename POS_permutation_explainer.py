import pickle
with open("pos_dictionary5k.pkl", "rb") as f:
    pos_dict = pickle.load(f)

print(pos_dict.keys())
