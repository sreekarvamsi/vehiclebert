# VehicleBERT – Domain-Specific NLP for Automotive Entities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/sreekarvamsi/vehiclebert.svg)](https://github.com/sreekarvamsi/vehiclebert/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/sreekarvamsi/vehiclebert.svg)](https://github.com/sreekarvamsi/vehiclebert/network)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Fine-tuned BERT model for Named Entity Recognition (NER) in automotive technical documentation. Extracts structured information from unstructured service manuals including vehicle parts, diagnostic codes, sensors, ECUs, and communication protocols.

## Key Features

- **High Accuracy**: 89% F1 score (vs 67% with generic NER models)
- **Fast Inference**: <50ms per document
- **Domain-Specific**: Trained on 5,000 annotated automotive sentences
- **12 Entity Types**: Comprehensive coverage of automotive domain
- **Production-Ready**: Includes training, inference, and deployment scripts

## 📊 Performance Metrics

| Metric | VehicleBERT | Generic BERT-NER |
|--------|-------------|------------------|
| F1 Score | **89%** | 67% |
| Precision | 91% | 70% |
| Recall | 87% | 64% |
| Inference Time | <50ms | ~45ms |

## Supported Entity Types

1. **VEHICLE_PART** - Physical components (engine, transmission, brake pad)
2. **DIAGNOSTIC_CODE** - OBD-II/manufacturer codes (P0420, B1234)
3. **SENSOR** - Electronic sensors (O2 sensor, MAP sensor)
4. **ECU** - Electronic Control Units (ECM, TCM, BCM)
5. **PROTOCOL** - Communication protocols (CAN bus, OBD-II, LIN)
6. **SYMPTOM** - Vehicle issues (rough idle, check engine light)
7. **TOOL** - Diagnostic tools (scan tool, multimeter)
8. **MEASUREMENT** - Specifications (5V, 2.5 bar, 90°C)
9. **PROCEDURE** - Service actions (bleed brakes, reset ECU)
10. **FLUID** - Vehicle fluids (engine oil, coolant, brake fluid)
11. **MANUFACTURER** - Vehicle makes (Toyota, Ford, BMW)
12. **MODEL** - Vehicle models (Camry, F-150, 3 Series)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vehiclebert.git
cd vehiclebert

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Inference

```python
from vehiclebert import VehicleBERTPredictor

# Load pre-trained model
predictor = VehicleBERTPredictor.from_pretrained("models/vehiclebert")

# Extract entities from text
text = "Replace the O2 sensor and clear the P0420 diagnostic code from the ECM."
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

### Training

```python
from vehiclebert import VehicleBERTTrainer

# Initialize trainer
trainer = VehicleBERTTrainer(
    model_name="bert-base-uncased",
    output_dir="models/vehiclebert"
)

# Train on your dataset
trainer.train(
    train_file="data/train.json",
    val_file="data/val.json",
    epochs=10,
    batch_size=16
)
```

## 📁 Project Structure

```
vehiclebert/
├── data/
│   ├── raw/                    # Raw automotive manuals
│   ├── annotated/              # Manually annotated data (5,000 sentences)
│   ├── train.json              # Training split (4,000 sentences)
│   ├── val.json                # Validation split (500 sentences)
│   └── test.json               # Test split (500 sentences)
├── models/
│   └── vehiclebert/            # Trained model checkpoints
├── src/
│   ├── __init__.py
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── model.py                # VehicleBERT model architecture
│   ├── trainer.py              # Training logic
│   ├── predictor.py            # Inference interface
│   └── evaluation.py           # Metrics and evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_annotation_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── scripts/
│   ├── prepare_data.py         # Data preparation pipeline
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── inference.py            # Inference script
├── tests/
│   ├── test_model.py
│   ├── test_predictor.py
│   └── test_data.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🎯 Applications

### 1. Automatic Manual Indexing
Automatically extract and index key entities from service manuals for quick lookup.

### 2. Knowledge Graph Construction
Build automotive knowledge graphs by extracting entities and their relationships.

### 3. RAG System Preprocessing
Enhance retrieval systems (like AutomotiveGPT) by extracting structured entities for better semantic search.

### 4. Smart Search
Enable entity-based search in technical documentation systems.

### 5. Quality Assurance
Validate technical documentation completeness by checking for expected entities.

## 📈 Training Details

- **Base Model**: `bert-base-uncased`
- **Training Data**: 5,000 manually annotated sentences (8,500 entity annotations)
- **Architecture**: Token classification head on BERT
- **Optimization**: AdamW optimizer with linear warmup
- **Hardware**: Trained on NVIDIA GPU (8GB+ VRAM recommended)
- **Training Time**: ~2 hours on single GPU

## 🔧 Advanced Usage

### Batch Processing

```python
from vehiclebert import VehicleBERTPredictor

predictor = VehicleBERTPredictor.from_pretrained("models/vehiclebert")

documents = [
    "Check the MAP sensor voltage at the ECM connector.",
    "Diagnostic code P0171 indicates a lean fuel mixture.",
    "Replace brake fluid according to manufacturer specifications."
]

results = predictor.predict_batch(documents, batch_size=8)
```

### Custom Entity Types

```python
# Add custom entity types to label mapping
custom_labels = {
    "CUSTOM_PART": 13,
    "CUSTOM_CODE": 14
}

predictor = VehicleBERTPredictor.from_pretrained(
    "models/vehiclebert",
    custom_labels=custom_labels
)
```

## 📊 Dataset Statistics

- **Total Sentences**: 5,000
- **Total Annotations**: 8,500
- **Average Entities/Sentence**: 1.7
- **Most Common Entity**: VEHICLE_PART (2,100 annotations)
- **Inter-Annotator Agreement**: 94% (Cohen's Kappa: 0.92)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use VehicleBERT in your research, please cite:

```bibtex
@software{vehiclebert2025,
  title={VehicleBERT: Domain-Specific NLP for Automotive Entities},
  author={Sreekar Vamsi Krishna Gajula},
  year={2025},
  url={https://github.com/sreekarvamsi/vehiclebert}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BERT model by Google Research
- Hugging Face Transformers library
- Automotive technical documentation sources

## 📧 Contact

Your Name - sreekarvamsikrishnag@gmail.com

Project Link: [https://github.com/sreekarvamsi/vehiclebert](https://github.com/sreekarvamsi/vehiclebert)

---

⭐ Star this repository if you find it helpful!
