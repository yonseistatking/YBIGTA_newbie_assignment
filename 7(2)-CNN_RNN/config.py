from typing import Literal


device = "cpu"
d_model = 256

# Word2Vec
window_size = 8
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 1e-03
num_epochs_word2vec = 9

# GRU
hidden_size = 256
num_classes = 4
lr = 6e-3
num_epochs = 100
batch_size = 16