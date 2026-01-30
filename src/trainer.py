"""
VehicleBERT Trainer
Handles model training, validation, and optimization
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from typing import Dict, List, Optional
from tqdm import tqdm
import time

from model import VehicleBERTModel, VehicleBERTConfig
from data_preparation import (
    AutomotiveNERDataset, 
    DatasetBuilder,
    AutomotiveSentence
)


class VehicleBERTTrainer:
    """Trainer class for VehicleBERT model"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dir: str = "models/vehiclebert",
        device: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: Base BERT model name or path
            output_dir: Directory to save trained model
            device: Device to use (cuda/cpu)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        print(f"Loading model: {model_name}")
        self.model = VehicleBERTModel(model_name)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Configuration
        self.config = VehicleBERTConfig()
        self.tokenizer = self.model.tokenizer
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.training_history = []
    
    def prepare_data(
        self,
        train_file: str,
        val_file: str,
        batch_size: int = 16,
        max_length: int = 128
    ) -> tuple:
        """
        Prepare data loaders.
        
        Args:
            train_file: Path to training data JSON
            val_file: Path to validation data JSON
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        print("Loading datasets...")
        
        # Load data
        train_sentences = DatasetBuilder.load_dataset(train_file)
        val_sentences = DatasetBuilder.load_dataset(val_file)
        
        print(f"Train sentences: {len(train_sentences)}")
        print(f"Val sentences: {len(val_sentences)}")
        
        # Create datasets
        train_dataset = AutomotiveNERDataset(
            train_sentences,
            self.tokenizer,
            self.config.ENTITY_LABELS,
            max_length
        )
        
        val_dataset = AutomotiveNERDataset(
            val_sentences,
            self.tokenizer,
            self.config.ENTITY_LABELS,
            max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training metrics
        """
        self.model.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        
        return {"loss": avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.model.eval()
        total_loss = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs["logits"], dim=-1)
                
                # Filter out padding tokens
                for i in range(len(predictions)):
                    mask = attention_mask[i] == 1
                    pred = predictions[i][mask].cpu().numpy()
                    label = labels[i][mask].cpu().numpy()
                    
                    all_predictions.extend(pred)
                    all_labels.extend(label)
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate accuracy
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        accuracy = correct / len(all_predictions)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def train(
        self,
        train_file: str,
        val_file: str,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        max_length: int = 128,
        save_best_only: bool = True
    ):
        """
        Train the model.
        
        Args:
            train_file: Path to training data
            val_file: Path to validation data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            max_length: Maximum sequence length
            save_best_only: Only save best model
        """
        print("\n" + "=" * 60)
        print("VehicleBERT Training")
        print("=" * 60)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            train_file, val_file, batch_size, max_length
        )
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            eps=1e-8
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        print(f"\nTraining for {epochs} epochs...")
        print(f"Total steps: {total_steps}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Print metrics
            print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics['loss'],
                "val_loss": val_metrics['loss'],
                "val_accuracy": val_metrics['accuracy']
            }
            self.training_history.append(epoch_metrics)
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                print(f"✓ New best validation loss: {self.best_val_loss:.4f}")
                
                if save_best_only:
                    self.save_checkpoint("best")
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"Training completed in {elapsed_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        # Save final model
        self.save_checkpoint("final")
        self.save_training_history()
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.save(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.output_dir, "training_history.json")
        
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    # Example usage
    trainer = VehicleBERTTrainer(
        model_name="bert-base-uncased",
        output_dir="models/vehiclebert"
    )
    
    # Check if data exists
    if not os.path.exists("data/train.json"):
        print("Creating sample dataset...")
        from data_preparation import create_sample_dataset
        create_sample_dataset()
    
    # Train model
    trainer.train(
        train_file="data/train.json",
        val_file="data/val.json",
        epochs=3,  # Use more epochs in production
        batch_size=16,
        learning_rate=5e-5
    )
