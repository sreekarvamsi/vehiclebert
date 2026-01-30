"""
VehicleBERT Model Architecture
Fine-tuned BERT for automotive Named Entity Recognition
"""

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from typing import Optional, Tuple, Dict, List


class VehicleBERTConfig:
    """Configuration class for VehicleBERT model"""
    
    # Entity type mapping
    ENTITY_LABELS = {
        "O": 0,  # Outside any entity
        "B-VEHICLE_PART": 1,
        "I-VEHICLE_PART": 2,
        "B-DIAGNOSTIC_CODE": 3,
        "I-DIAGNOSTIC_CODE": 4,
        "B-SENSOR": 5,
        "I-SENSOR": 6,
        "B-ECU": 7,
        "I-ECU": 8,
        "B-PROTOCOL": 9,
        "I-PROTOCOL": 10,
        "B-SYMPTOM": 11,
        "I-SYMPTOM": 12,
        "B-TOOL": 13,
        "I-TOOL": 14,
        "B-MEASUREMENT": 15,
        "I-MEASUREMENT": 16,
        "B-PROCEDURE": 17,
        "I-PROCEDURE": 18,
        "B-FLUID": 19,
        "I-FLUID": 20,
        "B-MANUFACTURER": 21,
        "I-MANUFACTURER": 22,
        "B-MODEL": 23,
        "I-MODEL": 24,
    }
    
    # Reverse mapping
    ID2LABEL = {v: k for k, v in ENTITY_LABELS.items()}
    
    # Simplified labels (without B-I prefix)
    SIMPLE_LABELS = [
        "VEHICLE_PART", "DIAGNOSTIC_CODE", "SENSOR", "ECU", 
        "PROTOCOL", "SYMPTOM", "TOOL", "MEASUREMENT",
        "PROCEDURE", "FLUID", "MANUFACTURER", "MODEL"
    ]
    
    def __init__(self):
        self.num_labels = len(self.ENTITY_LABELS)
        self.dropout = 0.1
        self.hidden_size = 768  # BERT base hidden size


class VehicleBERTForTokenClassification(BertPreTrainedModel):
    """
    VehicleBERT model for token classification (NER).
    Based on BERT with a token classification head.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # BERT encoder
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Dropout for regularization
        classifier_dropout = (
            config.classifier_dropout 
            if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Token classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Ground truth labels for training
            return_dict: Whether to return a dict
            
        Returns:
            Loss (if labels provided) and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        
        # Get sequence output (last hidden state)
        sequence_output = outputs[0]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Get logits from classification head
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None,
        }


class VehicleBERTModel:
    """
    Wrapper class for easier model usage.
    Handles model loading, saving, and provides convenience methods.
    """
    
    def __init__(self, model_path: str = "bert-base-uncased"):
        """
        Initialize VehicleBERT model.
        
        Args:
            model_path: Path to pretrained model or HuggingFace model name
        """
        from transformers import BertConfig, BertTokenizer
        
        self.config = VehicleBERTConfig()
        
        # Load BERT config and modify for token classification
        bert_config = BertConfig.from_pretrained(model_path)
        bert_config.num_labels = self.config.num_labels
        
        # Load model
        try:
            self.model = VehicleBERTForTokenClassification.from_pretrained(
                model_path, 
                config=bert_config
            )
        except:
            # If loading fails, initialize from scratch
            self.model = VehicleBERTForTokenClassification(bert_config)
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def save(self, output_dir: str):
        """Save model and tokenizer to directory"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        import json
        label_path = os.path.join(output_dir, "label_mapping.json")
        with open(label_path, "w") as f:
            json.dump({
                "label2id": self.config.ENTITY_LABELS,
                "id2label": self.config.ID2LABEL,
            }, f, indent=2)
    
    @classmethod
    def load(cls, model_path: str):
        """Load model from directory"""
        return cls(model_path)
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get model parameter counts"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }
    
    def freeze_bert_encoder(self):
        """Freeze BERT encoder layers (only train classification head)"""
        for param in self.model.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        """Unfreeze BERT encoder layers"""
        for param in self.model.bert.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Example usage and testing
    print("VehicleBERT Model Architecture")
    print("=" * 50)
    
    config = VehicleBERTConfig()
    print(f"Number of entity labels: {config.num_labels}")
    print(f"Entity types: {config.SIMPLE_LABELS}")
    
    # Initialize model
    model = VehicleBERTModel()
    params = model.get_num_parameters()
    
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # Test forward pass
    dummy_input = torch.randint(0, 1000, (2, 32)).to(model.device)
    dummy_mask = torch.ones((2, 32)).to(model.device)
    
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(dummy_input, attention_mask=dummy_mask)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {outputs['logits'].shape}")
    print(f"  Expected: (batch_size, sequence_length, num_labels)")
    
    print("\n✓ Model architecture loaded successfully!")
