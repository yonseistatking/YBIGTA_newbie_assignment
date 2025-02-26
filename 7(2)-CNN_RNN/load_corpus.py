from datasets import load_dataset
from transformers import AutoTokenizer

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    # Load the sentiment analysis dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize each text sample and join tokens as strings (avoiding padding tokens)
    for example in dataset["verse_text"]:
        tokens = tokenizer.tokenize(example)
        if tokens:  # Ignore empty tokenized examples
            corpus.append(" ".join(tokens))
    
    return corpus