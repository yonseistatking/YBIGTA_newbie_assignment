from typing import Literal


device = "cpu"
d_model = 256

# Word2Vec
window_size = 2
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 1e-03
num_epochs_word2vec = 3

# GRU
hidden_size = 512
num_classes = 4
lr = 1e-3
num_epochs = 300
batch_size = 16