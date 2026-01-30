"""
VehicleBERT - Domain-Specific NLP for Automotive Entities
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .model import VehicleBERTModel, VehicleBERTConfig, VehicleBERTForTokenClassification
from .predictor import VehicleBERTPredictor
from .trainer import VehicleBERTTrainer
from .evaluation import VehicleBERTEvaluator
from .data_preparation import (
    AutomotiveEntity,
    AutomotiveSentence,
    AutomotiveNERDataset,
    SyntheticDataGenerator,
    DatasetBuilder
)

__all__ = [
    "VehicleBERTModel",
    "VehicleBERTConfig",
    "VehicleBERTForTokenClassification",
    "VehicleBERTPredictor",
    "VehicleBERTTrainer",
    "VehicleBERTEvaluator",
    "AutomotiveEntity",
    "AutomotiveSentence",
    "AutomotiveNERDataset",
    "SyntheticDataGenerator",
    "DatasetBuilder",
]
