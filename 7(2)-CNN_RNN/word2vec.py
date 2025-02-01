import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal, List

# 구현하세요!


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
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        self.linear_out = nn.Linear(d_model, vocab_size, bias=False)
        # 구현하세요!
        pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        for epoch in range(num_epochs):
            total_loss = 0
            for sentence in corpus:
                tokens = tokenizer.convert_tokens_to_ids(sentence.split())
                contexts, targets = self._generate_training_pairs(tokens)
                
                for context, target in zip(contexts, targets):
                    optimizer.zero_grad()
                    if self.method == "cbow":
                        prediction = self._train_cbow(context)
                    else:
                        prediction = self._train_skipgram(target, context)

                    loss = criterion(prediction, target.unsqueeze(0))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
        pass

    def _train_cbow(
        self, context: Tensor
        # 구현하세요!
    ) -> Tensor:
        # 구현하세요!
        context_embeddings = self.embeddings(context).mean(dim=0).unsqueeze(0)
        return self.linear_out(context_embeddings)
    pass

    def _train_skipgram(
        self, target: Tensor, context: Tensor
        # 구현하세요!
    ) -> Tensor:
        # 구현하세요!
        output_loss = []
        for word in context:
            embedding = self.embeddings(word).unsqueeze(0)
            prediction = self.linear_out(embedding)
            output_loss.append(prediction)
        return torch.cat(output_loss, dim=0).mean(dim=0).unsqueeze(0)
    pass

    # 구현하세요!
    def _generate_training_pairs(self, tokens: List[int]):
        contexts, targets = [], []
        for i, token in enumerate(tokens):
            start = max(i - self.window_size, 0)
            end = min(i + self.window_size + 1, len(tokens))
            context = [tokens[j] for j in range(start, end) if j != i]
            if context:
                targets.append(torch.tensor(token))
                contexts.append(torch.tensor(context))
        return contexts, targets
    pass