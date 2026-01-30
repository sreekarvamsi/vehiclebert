#!/usr/bin/env python
"""
Data preparation script for VehicleBERT
Usage: python scripts/prepare_data.py [options]
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preparation import SyntheticDataGenerator, DatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Prepare data for VehicleBERT")
    
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=5000,
        help="Number of sentences to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for datasets"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        print("Error: Train, val, and test ratios must sum to 1.0")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "annotated"), exist_ok=True)
    
    print("=" * 60)
    print("VehicleBERT Data Preparation")
    print("=" * 60)
    print(f"\nGenerating {args.num_sentences} synthetic sentences...")
    
    # Generate data
    sentences = SyntheticDataGenerator.generate_dataset(args.num_sentences)
    
    # Create splits
    print(f"\nCreating splits:")
    print(f"  Train: {args.train_ratio:.1%}")
    print(f"  Val: {args.val_ratio:.1%}")
    print(f"  Test: {args.test_ratio:.1%}")
    
    train, val, test = DatasetBuilder.create_splits(
        sentences,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Save datasets
    print(f"\nSaving datasets to {args.output_dir}/...")
    
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    test_path = os.path.join(args.output_dir, "test.json")
    
    DatasetBuilder.save_dataset(train, train_path)
    DatasetBuilder.save_dataset(val, val_path)
    DatasetBuilder.save_dataset(test, test_path)
    
    # Calculate statistics
    total_entities = sum(len(s.entities) for s in sentences)
    avg_entities = total_entities / len(sentences)
    
    # Count entities by type
    entity_counts = {}
    for sentence in sentences:
        for entity in sentence.entities:
            entity_counts[entity.label] = entity_counts.get(entity.label, 0) + 1
    
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"\nTotal sentences: {len(sentences):,}")
    print(f"Total entities: {total_entities:,}")
    print(f"Average entities per sentence: {avg_entities:.2f}")
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train):,} sentences")
    print(f"  Val: {len(val):,} sentences")
    print(f"  Test: {len(test):,} sentences")
    
    print(f"\nEntity distribution:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_entities * 100
        print(f"  {entity_type:<20} {count:>6,} ({percentage:>5.1f}%)")
    
    # Show sample
    print(f"\nSample sentences:")
    print("-" * 60)
    for i, sentence in enumerate(sentences[:3], 1):
        print(f"\n{i}. {sentence.text}")
        print("   Entities:")
        for entity in sentence.entities:
            print(f"     - {entity.text} [{entity.label}]")
    
    print("\n" + "=" * 60)
    print("✓ Data preparation completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
