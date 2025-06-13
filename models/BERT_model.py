from transformers import BertTokenizer, BertModel
import torch
from models.model import Model

class BERT_model(Model):
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embeddings(self, sentence=None) -> list:
        if sentence is None:
            raise ValueError("Sentence cannot be None")
            
        tokenizer = self.tokenizer(sentence, return_tensors="pt")
        input_ids = tokenizer["input_ids"].to(self.device)
        attention_mask = tokenizer["attention_mask"].to(self.device)

        # Get model output
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the hidden states (last layer)
        last_hidden_state = outputs.last_hidden_state[0]

        # Convert to list (excluding special tokens [CLS] and [SEP])
        embeddings = last_hidden_state[1:-1].cpu().tolist()

        return embeddings

    def tokenize(self, sentence) -> list:
        return self.tokenizer.tokenize(sentence)

    def reconstruct_sentence(self, tokens) -> str:
        sentence = ""
        for token in tokens:
            if token.startswith("##"):
                sentence += token[2:] 
            else:
                if sentence:
                    sentence += " "
                sentence += token
        return sentence