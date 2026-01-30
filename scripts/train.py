#!/usr/bin/env python
"""
Training script for VehicleBERT
Usage: python scripts/train.py [options]
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer import VehicleBERTTrainer
from data_preparation import create_sample_dataset


def main():
    parser = argparse.ArgumentParser(description="Train VehicleBERT model")
    
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.json",
        help="Path to training data JSON file"
    )
    
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/val.json",
        help="Path to validation data JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/vehiclebert",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Base BERT model name or path"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--create-data",
        action="store_true",
        help="Create sample dataset if not exists"
    )
    
    args = parser.parse_args()
    
    # Create sample data if requested and doesn't exist
    if args.create_data or not os.path.exists(args.train_file):
        print("Creating sample dataset...")
        create_sample_dataset()
    
    # Check if data exists
    if not os.path.exists(args.train_file):
        print(f"Error: Training file not found: {args.train_file}")
        print("Run with --create-data to generate sample data")
        return 1
    
    if not os.path.exists(args.val_file):
        print(f"Error: Validation file not found: {args.val_file}")
        return 1
    
    # Initialize trainer
    print("\nInitializing VehicleBERT Trainer...")
    trainer = VehicleBERTTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train(
        train_file=args.train_file,
        val_file=args.val_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length
    )
    
    print("\n✓ Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
