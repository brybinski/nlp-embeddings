from transformers import BertTokenizer, BertModel
import torch
from models.model import Model
import numpy as np
class BERT_model(Model):
    def __init__(self, model_name="bert-base-uncased", sep_cls = False):
        self.model_name = model_name
        self.sep_cls = sep_cls
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
        if self.sep_cls:
            embeddings = last_hidden_state.cpu().tolist()
        embeddings = last_hidden_state[1:-1].cpu().tolist()

        return embeddings

    def tokenize(self, sentence) -> list:
        if self.sep_cls:
            token_list = self.tokenizer.tokenize(sentence)
            return ["[CLS]"] + token_list + ["[SEP]"]
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
    
    
    def tSNE_plot(self, sequences, n_components=2, save_path=None, show=False):
        """
        Generates a t-SNE plot for the embeddings of the given sequence.
        """
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))

        all_embeddings = []
        all_labels = []
        colors = []
        
        for n, sequence in enumerate(sequences):
            embeddings = self.get_embeddings(sequence)
            tokens = self.tokenize(sequence)
            all_embeddings.extend(embeddings)
            all_labels.extend(tokens)
            colors.extend([n] * len(embeddings))
            
    
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=7)

        reduced_embeddings = tsne.fit_transform(np.array(all_embeddings))

        
        
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
        for i, txt in enumerate(all_labels):
            plt.annotate(f"{txt}", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9)
        
        plt.title(f't-SNE plot of {self.model_name} embeddings for {len(sequences)} sequences')
        
        if save_path:
            plt.savefig(save_path, dpi=72, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
        