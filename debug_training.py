import os
import torch
from config import Config
from model import BERT
from dataset import create_dataloaders
from training import train_model

def debug_training():
    """Run a minimal training cycle to test the entire pipeline"""
    print("Starting debug training...")
    
    # Create a minimal config
    config = Config()
    config.max_examples = 100  # Very small dataset
    config.num_epochs = 1
    config.batch_size = 4
    config.max_sequence_length = 64
    config.num_hidden_layers = 2
    config.gradient_accumulation_steps = 1
    
    print("Creating dataloaders with minimal configuration...")
    try:
        train_dataloader, val_dataloader, tokenizer = create_dataloaders(config)
        print("Dataloaders created successfully.")
        
        print("Initializing minimal model...")
        model = BERT(config)
        print("Model initialized with parameters:", sum(p.numel() for p in model.parameters()))
        
        print("Running a single training epoch...")
        train_model(model, train_dataloader, val_dataloader, config)
        print("Debug training completed successfully!")
        return True
    except Exception as e:
        print(f"Debug training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_training()
