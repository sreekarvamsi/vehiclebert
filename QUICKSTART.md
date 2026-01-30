# VehicleBERT Quick Start Guide

Get started with VehicleBERT in 5 minutes!

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vehiclebert.git
cd vehiclebert

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 1. Prepare Data

Generate synthetic training data:

```bash
python scripts/prepare_data.py --num-sentences 5000
```

This creates:
- `data/train.json` (4,000 sentences)
- `data/val.json` (500 sentences)
- `data/test.json` (500 sentences)

## 2. Train Model

Train VehicleBERT on the dataset:

```bash
python scripts/train.py \
  --train-file data/train.json \
  --val-file data/val.json \
  --epochs 10 \
  --batch-size 16
```

**Note:** Training on CPU will be slow. Use GPU for faster training.

For quick testing (3 epochs):
```bash
python scripts/train.py --epochs 3
```

## 3. Run Inference

### Single Text

```bash
python scripts/inference.py \
  --model-path models/vehiclebert/best \
  --text "Replace the O2 sensor and clear the P0420 code from the ECM."
```

### Multiple Texts from File

```bash
# Create input file
echo "Check the MAP sensor voltage at the ECU connector." > input.txt
echo "The CAN bus communication error is causing issues." >> input.txt

# Run inference
python scripts/inference.py \
  --model-path models/vehiclebert/best \
  --input-file input.txt \
  --output-file results.json
```

### Benchmark Performance

```bash
python scripts/inference.py \
  --model-path models/vehiclebert/best \
  --benchmark
```

## 4. Evaluate Model

```bash
python scripts/evaluate.py \
  --model-path models/vehiclebert/best \
  --test-file data/test.json \
  --error-analysis
```

## Python API Usage

### Basic Inference

```python
from vehiclebert import VehicleBERTPredictor

# Load model
predictor = VehicleBERTPredictor.from_pretrained("models/vehiclebert/best")

# Predict entities
text = "Replace the O2 sensor and clear the P0420 code from the ECM."
entities = predictor.predict(text)

for entity in entities:
    print(f"{entity['text']} -> {entity['label']} ({entity['confidence']:.2f})")
```

**Output:**
```
O2 sensor -> SENSOR (0.96)
P0420 -> DIAGNOSTIC_CODE (0.94)
ECM -> ECU (0.98)
```

### Batch Processing

```python
texts = [
    "Check the MAP sensor voltage.",
    "Diagnostic code P0171 indicates a lean mixture.",
    "Inspect the brake fluid level."
]

results = predictor.predict_batch(texts, batch_size=8)

for text, entities in zip(texts, results):
    print(f"\n{text}")
    for entity in entities:
        print(f"  - {entity['text']} [{entity['label']}]")
```

### Training Custom Model

```python
from vehiclebert import VehicleBERTTrainer

# Initialize trainer
trainer = VehicleBERTTrainer(
    model_name="bert-base-uncased",
    output_dir="models/my_vehiclebert"
)

# Train
trainer.train(
    train_file="data/train.json",
    val_file="data/val.json",
    epochs=10,
    batch_size=16,
    learning_rate=5e-5
)
```

### Evaluation

```python
from vehiclebert import VehicleBERTPredictor, VehicleBERTEvaluator
from vehiclebert import DatasetBuilder

# Load model and test data
predictor = VehicleBERTPredictor.from_pretrained("models/vehiclebert/best")
test_sentences = DatasetBuilder.load_dataset("data/test.json")

# Evaluate
evaluator = VehicleBERTEvaluator(predictor)
metrics = evaluator.evaluate(test_sentences)

print(f"F1 Score: {metrics['f1']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
```

## Common Issues

### Out of Memory (GPU)

Reduce batch size:
```bash
python scripts/train.py --batch-size 8
```

### Slow Training (CPU)

Use fewer epochs for testing:
```bash
python scripts/train.py --epochs 3
```

Or use a smaller dataset:
```bash
python scripts/prepare_data.py --num-sentences 1000
```

### Import Errors

Make sure you're in the virtual environment:
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Next Steps

- 📖 Read the full [README.md](README.md)
- 🔬 Check out example notebooks in `notebooks/`
- 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- 📊 View detailed metrics in training history

## Expected Results

With default settings (5,000 sentences, 10 epochs):
- **F1 Score:** ~89%
- **Precision:** ~91%
- **Recall:** ~87%
- **Inference Time:** <50ms per document

## Support

- 🐛 [Report bugs](https://github.com/yourusername/vehiclebert/issues)
- 💡 [Request features](https://github.com/yourusername/vehiclebert/issues)
- 💬 [Ask questions](https://github.com/yourusername/vehiclebert/discussions)

Happy coding! 🚗
