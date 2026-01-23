from pathlib import Path
import torch
from pytorch_lightning import Trainer
from src.model import ECGClassifier
from src.data import ECGDataModule
import numpy as np


def test_training_step() -> None:
    """Test that training step works correctly."""
    model = ECGClassifier(lr=1e-3, num_classes=3)
    batch_size = 2
    x = torch.randn(batch_size, 1, 224, 224)
    y = torch.randint(0, 3, (batch_size,))
    
    loss = model.training_step((x, y), 0)
    
    assert loss is not None
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0

