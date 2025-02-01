import torch
from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset


from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *


if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint)
    embeddings = word2vec.embeddings_weight()

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load train, validation dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=True)

    # train
    for epoch in range(num_epochs):
        loss_sum = 0
        for data in train_loader:
            optimizer.zero_grad()
            input_ids = tokenizer(data["verse_text"], padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
            labels = data["label"].to(device)

            # Forward pass
            logits = model(input_ids)

            # If logits have extra sequence dimension, select the last time step
            if logits.ndim == 3:
                logits = logits[:, -1, :]  # [batch_size, num_classes]

            # Ensure labels have correct shape
            labels = labels.view(-1)  # [batch_size]

            # Compute loss
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        preds = []
        labels_list = []  # Avoid name collision
        with torch.no_grad():
            for data in validation_loader:
                input_ids = tokenizer(data["verse_text"], padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
                logits = model(input_ids)

                # If logits have extra sequence dimension, select the last time step
                if logits.ndim == 3:
                    logits = logits[:, -1, :]  # [batch_size, num_classes]

                labels_list += data["label"].tolist()
                preds += logits.argmax(-1).cpu().tolist()

        # Calculate F1 scores
        macro = f1_score(labels_list, preds, average='macro')
        micro = f1_score(labels_list, preds, average='micro')
        print(f"Epoch [{epoch+1}/{num_epochs}] | loss: {loss_sum/len(train_loader):.6f} | macro: {macro:.6f} | micro: {micro:.6f}")

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")
