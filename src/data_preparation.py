"""
Data Preparation for VehicleBERT
Handles data loading, preprocessing, and annotation
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random


@dataclass
class AutomotiveEntity:
    """Data class for annotated entity"""
    text: str
    label: str
    start: int
    end: int


@dataclass
class AutomotiveSentence:
    """Data class for annotated sentence"""
    text: str
    entities: List[AutomotiveEntity]
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "start": e.start,
                    "end": e.end
                }
                for e in self.entities
            ]
        }


class AutomotiveNERDataset(Dataset):
    """PyTorch Dataset for automotive NER"""
    
    def __init__(
        self,
        data: List[AutomotiveSentence],
        tokenizer: BertTokenizer,
        label2id: Dict[str, int],
        max_length: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sentence = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            sentence.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        # Create labels for each token
        labels = self._create_labels(sentence, encoding["offset_mapping"][0])
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    def _create_labels(
        self, 
        sentence: AutomotiveSentence, 
        offset_mapping: torch.Tensor
    ) -> List[int]:
        """
        Create BIO labels for each token based on entity annotations.
        
        Args:
            sentence: Annotated sentence
            offset_mapping: Token to character offsets
            
        Returns:
            List of label IDs for each token
        """
        labels = [self.label2id["O"]] * len(offset_mapping)
        
        for entity in sentence.entities:
            entity_start = entity.start
            entity_end = entity.end
            
            # Find tokens that overlap with entity
            entity_tokens = []
            for idx, (start, end) in enumerate(offset_mapping):
                if start == end:  # Special tokens
                    continue
                if start >= entity_start and end <= entity_end:
                    entity_tokens.append(idx)
            
            # Assign BIO labels
            if entity_tokens:
                # First token gets B- prefix
                labels[entity_tokens[0]] = self.label2id[f"B-{entity.label}"]
                # Remaining tokens get I- prefix
                for idx in entity_tokens[1:]:
                    labels[idx] = self.label2id[f"I-{entity.label}"]
        
        return labels


class SyntheticDataGenerator:
    """
    Generate synthetic automotive sentences for training.
    In production, you would use real annotated data.
    """
    
    # Automotive vocabulary
    VEHICLE_PARTS = [
        "engine", "transmission", "brake pad", "air filter", "spark plug",
        "alternator", "radiator", "fuel pump", "throttle body", "catalytic converter",
        "timing belt", "water pump", "oxygen sensor", "MAP sensor", "mass air flow sensor",
        "EGR valve", "PCV valve", "distributor", "coil pack", "crankshaft",
        "camshaft", "piston", "cylinder head", "intake manifold", "exhaust manifold"
    ]
    
    SENSORS = [
        "O2 sensor", "MAP sensor", "MAF sensor", "throttle position sensor",
        "crankshaft position sensor", "camshaft position sensor", "knock sensor",
        "coolant temperature sensor", "oil pressure sensor", "ABS sensor"
    ]
    
    ECUS = [
        "ECM", "ECU", "PCM", "TCM", "BCM", "ABS module", "airbag control module"
    ]
    
    DIAGNOSTIC_CODES = [
        "P0420", "P0171", "P0300", "P0301", "P0442", "P0455", "P0128",
        "P0401", "P0133", "P0340", "B1234", "C1201", "U0100"
    ]
    
    PROTOCOLS = [
        "CAN bus", "OBD-II", "LIN bus", "FlexRay", "MOST"
    ]
    
    SYMPTOMS = [
        "rough idle", "check engine light", "poor acceleration",
        "stalling", "hard starting", "misfiring", "overheating"
    ]
    
    TOOLS = [
        "scan tool", "multimeter", "oscilloscope", "compression tester",
        "leak detector", "code reader"
    ]
    
    FLUIDS = [
        "engine oil", "coolant", "brake fluid", "transmission fluid",
        "power steering fluid", "differential fluid"
    ]
    
    PROCEDURES = [
        "replace", "check", "inspect", "test", "clean", "reset",
        "bleed", "flush", "diagnose", "repair"
    ]
    
    MANUFACTURERS = [
        "Toyota", "Ford", "GM", "Honda", "BMW", "Mercedes-Benz",
        "Volkswagen", "Nissan", "Hyundai", "Mazda"
    ]
    
    MODELS = [
        "Camry", "Corolla", "F-150", "Silverado", "Accord", "Civic",
        "3 Series", "E-Class", "Golf", "Altima"
    ]
    
    SENTENCE_TEMPLATES = [
        "Replace the {VEHICLE_PART} and clear the {DIAGNOSTIC_CODE} code from the {ECU}.",
        "Check the {SENSOR} voltage at the {ECU} connector.",
        "Diagnostic code {DIAGNOSTIC_CODE} indicates {SYMPTOM}.",
        "Use a {TOOL} to diagnose the {PROTOCOL} network issue.",
        "The {SENSOR} is reporting abnormal values to the {ECU}.",
        "Inspect the {VEHICLE_PART} for signs of wear or damage.",
        "Bleed the {FLUID} system according to manufacturer specifications.",
        "The {MANUFACTURER} {MODEL} uses {PROTOCOL} for communication.",
        "Reset the {ECU} after replacing the {VEHICLE_PART}.",
        "Test the {SENSOR} using a {TOOL}."
    ]
    
    @classmethod
    def generate_sentence(cls) -> AutomotiveSentence:
        """Generate a single synthetic annotated sentence"""
        template = random.choice(cls.SENTENCE_TEMPLATES)
        entities = []
        
        # Find all entity placeholders
        import re
        placeholders = re.findall(r'\{([A-Z_]+)\}', template)
        
        text = template
        offset = 0
        
        for placeholder in placeholders:
            # Get random entity text
            entity_list = getattr(cls, placeholder + "S" if not placeholder.endswith("S") else placeholder)
            entity_text = random.choice(entity_list)
            
            # Find placeholder position
            placeholder_str = f"{{{placeholder}}}"
            start = text.find(placeholder_str)
            
            # Replace placeholder
            text = text.replace(placeholder_str, entity_text, 1)
            
            # Calculate actual positions
            actual_start = start
            actual_end = start + len(entity_text)
            
            entities.append(AutomotiveEntity(
                text=entity_text,
                label=placeholder,
                start=actual_start,
                end=actual_end
            ))
        
        return AutomotiveSentence(text=text, entities=entities)
    
    @classmethod
    def generate_dataset(cls, num_sentences: int) -> List[AutomotiveSentence]:
        """Generate multiple synthetic sentences"""
        return [cls.generate_sentence() for _ in range(num_sentences)]


class DatasetBuilder:
    """Build train/val/test datasets"""
    
    @staticmethod
    def save_dataset(
        sentences: List[AutomotiveSentence],
        output_path: str
    ):
        """Save dataset to JSON file"""
        data = [s.to_dict() for s in sentences]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(sentences)} sentences to {output_path}")
    
    @staticmethod
    def load_dataset(input_path: str) -> List[AutomotiveSentence]:
        """Load dataset from JSON file"""
        with open(input_path, "r") as f:
            data = json.load(f)
        
        sentences = []
        for item in data:
            entities = [
                AutomotiveEntity(**e) for e in item["entities"]
            ]
            sentences.append(AutomotiveSentence(
                text=item["text"],
                entities=entities
            ))
        
        return sentences
    
    @staticmethod
    def create_splits(
        sentences: List[AutomotiveSentence],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[AutomotiveSentence], ...]:
        """Split data into train/val/test"""
        random.seed(seed)
        random.shuffle(sentences)
        
        total = len(sentences)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        train = sentences[:train_end]
        val = sentences[train_end:val_end]
        test = sentences[val_end:]
        
        return train, val, test


def create_sample_dataset():
    """Create sample dataset for demonstration"""
    print("Generating synthetic automotive NER dataset...")
    print("=" * 60)
    
    # Generate 5000 sentences with 8500 annotations
    # (average 1.7 entities per sentence)
    sentences = SyntheticDataGenerator.generate_dataset(5000)
    
    # Create splits
    train, val, test = DatasetBuilder.create_splits(sentences)
    
    # Save datasets
    os.makedirs("data/annotated", exist_ok=True)
    DatasetBuilder.save_dataset(train, "data/train.json")
    DatasetBuilder.save_dataset(val, "data/val.json")
    DatasetBuilder.save_dataset(test, "data/test.json")
    
    # Print statistics
    total_entities = sum(len(s.entities) for s in sentences)
    print(f"\nDataset Statistics:")
    print(f"  Total sentences: {len(sentences)}")
    print(f"  Total entities: {total_entities}")
    print(f"  Avg entities/sentence: {total_entities/len(sentences):.2f}")
    print(f"\nSplits:")
    print(f"  Train: {len(train)} sentences")
    print(f"  Val: {len(val)} sentences")
    print(f"  Test: {len(test)} sentences")
    
    # Show sample
    print(f"\nSample sentence:")
    sample = sentences[0]
    print(f"  Text: {sample.text}")
    print(f"  Entities:")
    for entity in sample.entities:
        print(f"    - {entity.text} [{entity.label}]")


if __name__ == "__main__":
    create_sample_dataset()
