import os
import torch
from datasets import load_dataset
from config import Config
from tokenizer import WordPieceTokenizer

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def debug_tokenizer():
    """Function to debug tokenizer output"""
    config = Config()

    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset(
        config.dataset_name, config.dataset_subset, trust_remote_code=True
    )

    # Print dataset info
    print(f"Dataset loaded: {dataset}")
    print(f"Splits: {list(dataset.keys())}")

    # Get a sample text
    sample = dataset["full"][0]
    sample_text = sample["text"]
    print(f"\nSample text:\n{sample_text[:200]}...\n")

    # Try loading the tokenizer
    print("Loading tokenizer...")
    tokenizer_handler = WordPieceTokenizer(config)
    try:
        tokenizer = tokenizer_handler.load()
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Training a new tokenizer...")
        tokenizer = tokenizer_handler.train([sample_text])

    # Test tokenization directly with the HuggingFace tokenizer
    print("\nTokenizing with HuggingFace tokenizer...")
    tokens = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=config.max_sequence_length,
        return_tensors="pt",
    )
    print(f"Tokenizer output type: {type(tokens)}")
    print(f"Tokenizer output keys: {tokens.keys()}")
    for key, value in tokens.items():
        print(f"{key}: {type(value)}, shape: {value.shape}")

    # Test if tokens can be moved to device
    if torch.cuda.is_available():
        print("\nMoving tokens to CUDA...")
        for key, value in tokens.items():
            if hasattr(value, "to"):
                cuda_value = value.to("cuda")
                print(f"{key} moved to CUDA: {cuda_value.device}")
            else:
                print(f"{key} cannot be moved to CUDA")

    # Return successful
    return True


if __name__ == "__main__":
    result = debug_tokenizer()
    print(f"\nDebug completed: {'Successfully' if result else 'Failed'}")
