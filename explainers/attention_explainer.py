from explainers.explainer import Explainer
from models.model import Model
from explainers.explanation import Explanation
import torch
import numpy as np

class BertAttentionExplainer(Explainer):
    """
    Explains token embeddings by analyzing the multi-head attention weights 
    across all attention layers in BERT-like models.
    """
    
    def __init__(self, model: Model, **kwargs):
        self.model = model
        
        self.include_cls_sep = kwargs.get("include_cls_sep", False)
        self.aggregation_method = kwargs.get("aggregation_method", "sum")  # Options: sum, mean, max
        
    def explainEmbeddings(self, sentence, word_idx=None, **kwargs) -> Explanation:
        """
        Generate explanation for token embeddings based on attention weights.
        
        Args:
            sentence: Input sentence to explain
            word_idx: Optional index of specific token to explain (None means explain all)
            
        Returns:
            Explanation object with attention-based influence scores
        """
        # Get tokens and check if they match the model's tokenization
        tokens = self.model.tokenize(sentence)
        
        # Get attention weights from the modelrt
        attention_weights = self._get_attention_weights(sentence)
        
        # Create explanation object
        explanation = Explanation("attention", sentence, tokens)
        
        # Process attention weights and add to explanation
        if word_idx is not None:
            # Explain only the specified token
            print(f"Explaining token: \"{tokens[word_idx]}\"")
            self._process_token_attention(tokens, attention_weights, explanation, word_idx)
        else:
            # Explain all tokens
            for idx in range(len(tokens)):
                print(f"Token \"{tokens[idx]}\": {idx+1}/{len(tokens)}")
                self._process_token_attention(tokens, attention_weights, explanation, idx)
        
        return explanation
    
    def _get_attention_weights(self, sentence):
        """
        Extract attention weights from the model for the given sentence.
        """
        # Tokenize input
        inputs = self.model.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Run model with output_attentions=True to get attention weights
        self.model.model.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
        # Get attention tensors (shape: [layers, batch, heads, seq_len, seq_len])
        attention_weights = outputs.attentions
        
        return attention_weights
    
    def _process_token_attention(self, tokens, attention_weights, explanation, token_idx):
        """
        Process attention weights for a specific token and add to explanation.
        
        Args:
            tokens: List of tokens
            attention_weights: Attention weights from the model
            explanation: Explanation object to update
            token_idx: Index of the token to explain
        """
        # Convert attention_weights to numpy arrays for easier manipulation
        attn_arrays = [attn_layer.cpu().numpy()[0] for attn_layer in attention_weights]
        
        # Aggregate attention weights across all layers and heads
        aggregated_attention = self._aggregate_attention_weights(attn_arrays)
        
        # Adjust index for CLS token (BERT adds [CLS] at beginning)
        bert_token_idx = token_idx + 1
        
        # Get attention scores for this token (how much it attends to other tokens)
        token_attention = aggregated_attention[bert_token_idx, 1:-1]  # Exclude [CLS] and [SEP]
        
        # Add scores to explanation object
        for i, score in enumerate(token_attention):
            explanation.add_one_word(
                tokens[token_idx],  # Main token
                token_idx,          # Main position
                tokens[i],          # Sub token 
                i,                  # Sub position
                float(score)        # Attention score
            )
    
    def _aggregate_attention_weights(self, attention_arrays):
        """
        Aggregate attention weights across all layers and heads.
        
        Args:
            attention_arrays: List of attention weight arrays per layer
            
        Returns:
            Aggregated attention weights as numpy array
        """
        # Stack all layers
        all_layers = np.stack(attention_arrays)
        
        # Aggregate across layers and heads
        if self.aggregation_method == "sum":
            # Sum across all layers and heads
            return np.sum(all_layers, axis=(0, 1))
        elif self.aggregation_method == "mean":
            # Average across all layers and heads
            return np.mean(all_layers, axis=(0, 1))
        elif self.aggregation_method == "max":
            # Take maximum attention score across all layers and heads
            return np.max(all_layers, axis=(0, 1))
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")