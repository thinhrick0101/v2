import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import gc
from config import Config
from model import BERT


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_model(model, train_dataloader, val_dataloader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # Print GPU memory info
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        )
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        # Set memory optimization
        torch.cuda.empty_cache()
        gc.collect()

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    # Adjust steps for gradient accumulation
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    total_steps = (
        len(train_dataloader) // config.gradient_accumulation_steps
    ) * config.num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # Training metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0

    # Create directories for saving models
    os.makedirs(config.model_save_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch

        # Training loop
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Calculate loss and scale by gradient accumulation steps
                loss = criterion(outputs, labels) / config.gradient_accumulation_steps
                epoch_loss += loss.item() * config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

                # Update progress bar
                progress_bar.set_postfix(
                    {"loss": loss.item() * config.gradient_accumulation_steps}
                )

                # Gradient accumulation: only update every config.gradient_accumulation_steps
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Free up memory
                    del input_ids, attention_mask, labels, outputs, loss, preds
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate epoch metrics
        epoch_loss = epoch_loss / len(train_dataloader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, F1: {epoch_f1:.4f}"
        )

        # Validation
        val_loss, val_accuracy, val_f1 = evaluate_model(
            model, val_dataloader, criterion, device, config
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}"
        )

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                os.path.join(config.model_save_dir, f"bert_star_rating_best.pt"),
            )

        # Save the model checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
            },
            os.path.join(config.model_save_dir, f"bert_star_rating_checkpoint.pt"),
        )

        # Clean up memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Plot training and validation metrics
    try:
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, config)
    except Exception as e:
        print(f"Error plotting metrics: {e}")

    return model


def evaluate_model(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0  # Keep track of processed batches

    with torch.no_grad():
        for batch in dataloader:
            try:
                # Process in smaller chunks to avoid OOM
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1

                # Clean up GPU memory
                del input_ids, attention_mask, labels, outputs, loss, preds
                if device.type == "cuda" and batch_count % 10 == 0:  # Every 10 batches
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue

    avg_loss = total_loss / max(batch_count, 1)  # Avoid division by zero

    if len(all_labels) > 0 and len(all_preds) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
    else:
        print("Warning: No predictions or labels available for evaluation")
        accuracy = 0.0
        f1 = 0.0

    return avg_loss, accuracy, f1


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, config):
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, "training_metrics.png"))
    plt.close()
