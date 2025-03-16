class Config:
    # Dataset configuration
    dataset_name = "nanaut/MLVU144"
    dataset_subset = "raw_review_All_Beauty"
    max_examples = 300000  # Limit number of examples to process

    # Tokenizer configuration
    vocab_size = 30522
    max_sequence_length = 512  # Reduced from 512 to save memory
    mask_prob = 0.15

    # Model configuration
    hidden_size = 768
    num_hidden_layers = 8  # Reduced from 12 to save memory
    num_attention_heads = 12
    intermediate_size = 3072
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1

    # Training configuration
    batch_size = 32  # Reduced from 32 to save memory
    num_epochs = 10
    learning_rate = 2e-5
    warmup_steps = 10000
    weight_decay = 0.01
    gradient_accumulation_steps = 4  # Added gradient accumulation to save memory
    max_grad_norm = 1.0

    # Output
    num_classes = 5  # Star ratings 1-5

    # Paths
    tokenizer_dir = "e:/Project/v2/tokenizer"
    model_save_dir = "e:/Project/v2/saved_models"
    log_dir = "e:/Project/v2/logs"
