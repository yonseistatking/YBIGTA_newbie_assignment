import torch
from torch import nn, Tensor
from torch.optim import Adam
import random

from transformers import PreTrainedTokenizer
from typing import Literal

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
        self.context_window = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        loss_function = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        tokenized_corpus = [tokenizer.tokenize(sentence) for sentence in corpus]
        tokenized_corpus = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_corpus]

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for sentence in tokenized_corpus:
                if len(sentence) < 2 * self.context_window + 1:
                    continue
                if self.method == "cbow":
                    loss = self._train_cbow(sentence, loss_function, optimizer)
                else:
                    loss = self._train_skipgram(sentence, loss_function, optimizer)
                epoch_loss += loss
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    def _train_cbow(
        self,
        sentence: list[int],
        loss_function: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        total_loss = 0
        for i in range(self.context_window, len(sentence) - self.context_window):
            context_tokens = sentence[i - self.context_window:i] + sentence[i+1:i+self.context_window+1]
            target_token = sentence[i]

            context_tensor = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor(target_token, dtype=torch.long).unsqueeze(0)
            context_embedding = self.embeddings(context_tensor).mean(dim=1)
            predictions = self.projection(context_embedding)
            loss_value = loss_function(predictions, target_tensor)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.item()
        return total_loss

    def _train_skipgram(
        self,
        sentence: list[int],
        loss_function: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        total_loss = 0
        for i in range(self.context_window, len(sentence) - self.context_window):
            target_token = sentence[i]
            context_tokens = sentence[i - self.context_window:i] + sentence[i+1:i+self.context_window+1]
            
            target_tensor = torch.tensor(target_token, dtype=torch.long).to(self.embeddings.weight.device)
            context_tensor = torch.tensor(context_tokens, dtype=torch.long).to(self.embeddings.weight.device)
            
            target_embedding = self.embeddings(target_tensor)  # (d_model,)
            predictions = self.projection(target_embedding)  # (vocab_size,)
            
            
            predictions = predictions.repeat(len(context_tokens), 1)  # (context_size, vocab_size)
            
            loss_value = loss_function(predictions, context_tensor)  # (context_size, vocab_size) vs. (context_size,)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.item()
        return total_loss

