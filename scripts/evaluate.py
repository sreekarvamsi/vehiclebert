#!/usr/bin/env python
"""
Evaluation script for VehicleBERT
Usage: python scripts/evaluate.py --model-path models/vehiclebert --test-file data/test.json
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import VehicleBERTPredictor
from evaluation import VehicleBERTEvaluator
from data_preparation import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Evaluate VehicleBERT model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/test.json",
        help="Path to test data JSON file"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for evaluation results (JSON)"
    )
    
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Perform error analysis"
    )
    
    parser.add_argument(
        "--num-errors",
        type=int,
        default=10,
        help="Number of errors to analyze"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return 1
    
    # Check if test file exists
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        return 1
    
    # Load predictor
    print(f"Loading model from {args.model_path}...")
    predictor = VehicleBERTPredictor.from_pretrained(args.model_path)
    print("✓ Model loaded successfully!")
    
    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    test_sentences = DatasetBuilder.load_dataset(args.test_file)
    print(f"✓ Loaded {len(test_sentences)} test sentences")
    
    # Initialize evaluator
    evaluator = VehicleBERTEvaluator(predictor)
    
    # Evaluate
    metrics = evaluator.evaluate(test_sentences, verbose=True)
    
    # Save results
    if args.output_file:
        evaluator.save_evaluation(metrics, args.output_file)
    
    # Error analysis
    if args.error_analysis:
        print("\n" + "=" * 60)
        print("Error Analysis")
        print("=" * 60)
        
        errors = evaluator.error_analysis(test_sentences, args.num_errors)
        
        print(f"\nFound {len(errors)} errors. Showing first {args.num_errors}:\n")
        
        for i, error in enumerate(errors, 1):
            print(f"Example {i}:")
            print(f"Text: {error['text']}")
            
            if error['missed']:
                print("Missed entities:")
                for label, start, end in error['missed']:
                    print(f"  - {error['text'][start:end]} [{label}]")
            
            if error['extra']:
                print("Incorrectly predicted:")
                for label, start, end in error['extra']:
                    print(f"  - {error['text'][start:end]} [{label}]")
            
            print()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Overall F1 Score: {metrics['f1']:.2%}")
    print(f"Overall Precision: {metrics['precision']:.2%}")
    print(f"Overall Recall: {metrics['recall']:.2%}")
    print("\n✓ Evaluation completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
