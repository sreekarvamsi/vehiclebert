"""
Unit tests for VehicleBERT model
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import VehicleBERTModel, VehicleBERTConfig


class TestVehicleBERTConfig:
    """Test VehicleBERTConfig class"""
    
    def test_config_labels(self):
        config = VehicleBERTConfig()
        
        # Check number of labels
        assert config.num_labels == 25  # 12 entity types * 2 (B-/I-) + 1 (O)
        
        # Check O label exists
        assert "O" in config.ENTITY_LABELS
        assert config.ENTITY_LABELS["O"] == 0
        
        # Check B- and I- prefixes exist
        assert "B-VEHICLE_PART" in config.ENTITY_LABELS
        assert "I-VEHICLE_PART" in config.ENTITY_LABELS
    
    def test_label_mapping_consistency(self):
        config = VehicleBERTConfig()
        
        # Check that ID2LABEL is reverse of ENTITY_LABELS
        for label, idx in config.ENTITY_LABELS.items():
            assert config.ID2LABEL[idx] == label
    
    def test_simple_labels(self):
        config = VehicleBERTConfig()
        
        # Check simple labels are correct
        expected_labels = [
            "VEHICLE_PART", "DIAGNOSTIC_CODE", "SENSOR", "ECU",
            "PROTOCOL", "SYMPTOM", "TOOL", "MEASUREMENT",
            "PROCEDURE", "FLUID", "MANUFACTURER", "MODEL"
        ]
        
        assert config.SIMPLE_LABELS == expected_labels


class TestVehicleBERTModel:
    """Test VehicleBERTModel class"""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing"""
        return VehicleBERTModel("bert-base-uncased")
    
    def test_model_initialization(self, model):
        """Test model initializes correctly"""
        assert model.model is not None
        assert model.tokenizer is not None
        assert isinstance(model.config, VehicleBERTConfig)
    
    def test_model_device(self, model):
        """Test model is on correct device"""
        assert model.device in [torch.device("cuda"), torch.device("cpu")]
    
    def test_model_parameters(self, model):
        """Test model has parameters"""
        params = model.get_num_parameters()
        
        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params
        
        assert params["total"] > 0
        assert params["trainable"] > 0
    
    def test_forward_pass(self, model):
        """Test model forward pass"""
        # Create dummy input
        batch_size = 2
        seq_length = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        
        # Move to device
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        
        # Forward pass
        model.model.eval()
        with torch.no_grad():
            outputs = model.model(input_ids, attention_mask=attention_mask)
        
        # Check output shape
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_length, model.config.num_labels)
    
    def test_freeze_unfreeze(self, model):
        """Test freezing and unfreezing encoder"""
        # Initially all parameters should be trainable
        initial_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # Freeze encoder
        model.freeze_bert_encoder()
        frozen_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # Should have fewer trainable parameters
        assert frozen_trainable < initial_trainable
        
        # Unfreeze
        model.unfreeze_bert_encoder()
        unfrozen_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # Should match initial
        assert unfrozen_trainable == initial_trainable


class TestModelSaveLoad:
    """Test model saving and loading"""
    
    def test_save_and_load(self, tmp_path):
        """Test model can be saved and loaded"""
        # Create and save model
        model1 = VehicleBERTModel("bert-base-uncased")
        output_dir = tmp_path / "test_model"
        model1.save(str(output_dir))
        
        # Check files exist
        assert (output_dir / "config.json").exists()
        assert (output_dir / "pytorch_model.bin").exists()
        assert (output_dir / "label_mapping.json").exists()
        
        # Load model
        model2 = VehicleBERTModel.load(str(output_dir))
        
        # Check loaded model works
        assert model2.model is not None
        assert model2.tokenizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
