#!/usr/bin/env python
"""
Inference script for VehicleBERT
Usage: python scripts/inference.py --model-path models/vehiclebert --text "Your text here"
"""

import sys
import os
import argparse
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import VehicleBERTPredictor


def main():
    parser = argparse.ArgumentParser(description="Run VehicleBERT inference")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        help="Text to analyze"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        help="File containing texts (one per line)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing multiple texts"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Train a model first using scripts/train.py")
        return 1
    
    # Load predictor
    print(f"Loading model from {args.model_path}...")
    try:
        predictor = VehicleBERTPredictor.from_pretrained(
            args.model_path,
            confidence_threshold=args.confidence_threshold
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Process single text
    if args.text:
        print("\n" + "=" * 60)
        print("Analyzing text...")
        print("=" * 60)
        
        analysis = predictor.analyze_text(args.text, verbose=True)
        
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\n✓ Results saved to {args.output_file}")
    
    # Process file
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            return 1
        
        print(f"\nProcessing file: {args.input_file}")
        
        with open(args.input_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(texts)} texts to process")
        
        results = predictor.predict_batch(
            texts,
            batch_size=args.batch_size
        )
        
        # Format results
        output = [
            {
                "text": text,
                "entities": entities
            }
            for text, entities in zip(texts, results)
        ]
        
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"✓ Results saved to {args.output_file}")
        else:
            # Print to console
            for item in output:
                print(f"\nText: {item['text']}")
                print("Entities:")
                for entity in item['entities']:
                    print(f"  - {entity['text']} [{entity['label']}]")
    
    # Run benchmark
    elif args.benchmark:
        sample_texts = [
            "Replace the O2 sensor and clear the P0420 diagnostic code from the ECM.",
            "Check the MAP sensor voltage at the ECU connector.",
            "The CAN bus communication error is causing issues."
        ]
        
        print("\n" + "=" * 60)
        print("Running benchmark...")
        print("=" * 60)
        
        stats = predictor.benchmark(sample_texts, num_runs=100)
        
        print(f"\nBenchmark Results (100 runs):")
        print(f"  Average time: {stats['avg_time_ms']:.2f} ms")
        print(f"  Min time: {stats['min_time_ms']:.2f} ms")
        print(f"  Max time: {stats['max_time_ms']:.2f} ms")
        print(f"  Median time: {stats['median_time_ms']:.2f} ms")
        print("\n✓ Target: <50ms per inference")
        
        if stats['avg_time_ms'] < 50:
            print("✓ Performance target met!")
        else:
            print("⚠ Performance target not met")
    
    else:
        parser.print_help()
        print("\nError: Provide --text, --input-file, or --benchmark")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
