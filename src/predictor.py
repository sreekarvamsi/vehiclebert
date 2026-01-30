"""
VehicleBERT Predictor
Inference interface for trained model
"""

import torch
from typing import List, Dict, Tuple, Optional
import time
from transformers import BertTokenizer

from model import VehicleBERTModel, VehicleBERTConfig


class VehicleBERTPredictor:
    """Predictor class for VehicleBERT inference"""
    
    def __init__(
        self,
        model: VehicleBERTModel,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained VehicleBERT model
            confidence_threshold: Minimum confidence for predictions
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.config = VehicleBERTConfig()
        self.confidence_threshold = confidence_threshold
        
        # Set model to eval mode
        self.model.model.eval()
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load predictor from pretrained model.
        
        Args:
            model_path: Path to model directory
            **kwargs: Additional arguments for predictor
            
        Returns:
            VehicleBERTPredictor instance
        """
        model = VehicleBERTModel.load(model_path)
        return cls(model, **kwargs)
    
    def predict(
        self,
        text: str,
        return_confidence: bool = True
    ) -> List[Dict[str, any]]:
        """
        Predict entities in text.
        
        Args:
            text: Input text
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of entity dictionaries
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)
        offset_mapping = encoding["offset_mapping"][0]
        
        # Predict
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get predictions and confidences
        logits = outputs["logits"][0]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values
        
        # Convert to entities
        entities = self._extract_entities(
            text,
            predictions.cpu().numpy(),
            confidences.cpu().numpy(),
            offset_mapping,
            return_confidence
        )
        
        return entities
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        return_confidence: bool = True
    ) -> List[List[Dict[str, any]]]:
        """
        Predict entities for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of entity lists
        """
        all_entities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_entities = [
                self.predict(text, return_confidence)
                for text in batch_texts
            ]
            all_entities.extend(batch_entities)
        
        return all_entities
    
    def _extract_entities(
        self,
        text: str,
        predictions: list,
        confidences: list,
        offset_mapping: torch.Tensor,
        return_confidence: bool
    ) -> List[Dict[str, any]]:
        """
        Extract entities from predictions using BIO scheme.
        
        Args:
            text: Original text
            predictions: Predicted label IDs
            confidences: Confidence scores
            offset_mapping: Token to character offsets
            return_confidence: Whether to include confidence
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        current_entity = None
        
        for idx, (pred_id, conf, (start, end)) in enumerate(
            zip(predictions, confidences, offset_mapping)
        ):
            # Skip special tokens
            if start == end:
                continue
            
            # Get label
            label = self.config.ID2LABEL[pred_id]
            
            # Skip "O" (outside) labels
            if label == "O":
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Check confidence threshold
            if conf < self.confidence_threshold:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Parse BIO label
            bio_tag = label[0]  # B or I
            entity_type = label[2:]  # Entity type without B-/I-
            
            if bio_tag == "B":
                # Begin new entity
                if current_entity is not None:
                    entities.append(current_entity)
                
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": int(start),
                    "end": int(end),
                    "confidence": float(conf) if return_confidence else None
                }
            
            elif bio_tag == "I":
                # Continue current entity
                if current_entity is not None and current_entity["label"] == entity_type:
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["end"] = int(end)
                    # Update confidence (average)
                    if return_confidence:
                        current_entity["confidence"] = (
                            current_entity["confidence"] + float(conf)
                        ) / 2
        
        # Add last entity
        if current_entity is not None:
            entities.append(current_entity)
        
        # Remove confidence if not requested
        if not return_confidence:
            for entity in entities:
                entity.pop("confidence", None)
        
        return entities
    
    def benchmark(
        self,
        texts: List[str],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            texts: Sample texts for benchmarking
            num_runs: Number of runs
            
        Returns:
            Benchmark statistics
        """
        print(f"Running benchmark with {num_runs} runs...")
        
        times = []
        
        for _ in range(num_runs):
            text = texts[0]  # Use first text
            
            start_time = time.time()
            self.predict(text)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "median_time_ms": sorted(times)[len(times) // 2]
        }
    
    def analyze_text(
        self,
        text: str,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Analyze text and return structured information.
        
        Args:
            text: Input text
            verbose: Print detailed output
            
        Returns:
            Analysis dictionary
        """
        entities = self.predict(text)
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            label = entity["label"]
            if label not in entities_by_type:
                entities_by_type[label] = []
            entities_by_type[label].append(entity)
        
        analysis = {
            "text": text,
            "total_entities": len(entities),
            "entities": entities,
            "entities_by_type": entities_by_type
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("VehicleBERT Analysis")
            print("=" * 60)
            print(f"\nText: {text}\n")
            print(f"Total entities found: {len(entities)}\n")
            
            for entity_type, entity_list in entities_by_type.items():
                print(f"{entity_type}:")
                for e in entity_list:
                    conf_str = f" (confidence: {e.get('confidence', 0):.2f})" if 'confidence' in e else ""
                    print(f"  - {e['text']}{conf_str}")
            print("=" * 60)
        
        return analysis


def demo():
    """Demonstration of VehicleBERT predictor"""
    
    # Example texts
    example_texts = [
        "Replace the O2 sensor and clear the P0420 diagnostic code from the ECM.",
        "Check the MAP sensor voltage at the ECU connector using a multimeter.",
        "Diagnostic code P0171 indicates a lean fuel mixture in the Toyota Camry.",
        "Inspect the brake fluid level and bleed the brake system if necessary.",
        "The CAN bus communication error is causing the check engine light to illuminate.",
        "Use a scan tool to diagnose the ABS module on the Ford F-150.",
    ]
    
    print("\n" + "=" * 60)
    print("VehicleBERT Predictor Demo")
    print("=" * 60)
    print("\nNote: This demo requires a trained model.")
    print("Run the trainer first to create a model, or load a pretrained model.")
    print("\nExample usage:")
    print("-" * 60)
    
    for text in example_texts[:3]:
        print(f"\nText: {text}")
        print("Entities: [would extract entities here with trained model]")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
