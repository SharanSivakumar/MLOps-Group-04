import pytest
import torch

from src.model import ECGClassifier


def test_model():
    """Test the ECGClassifier initialization."""
    model = ECGClassifier(lr=0.001, num_classes=3)
    x = torch.randn(1, 1, 224, 224)  # ECG data: 1 channel, 224x224
    y = model(x)
    assert len(y.shape) == 2, "Output should be 2D (batch_size, num_classes)"
    assert y.shape[0] == 1, "Batch size should be 1"
    assert y.shape[1] == 3, "Should have 3 output classes (AF, Noise, NSR)"


def test_error_on_wrong_shape():
    model = ECGClassifier(lr=0.001, num_classes=3)
    with pytest.raises((ValueError, RuntimeError)):
        model(torch.randn(1,2,3))  # Wrong number of dimensions
    with pytest.raises((ValueError, RuntimeError)):
        model(torch.randn(1,1,28,28))  # Wrong spatial dimensions for EfficientNet


# try different input types
@pytest.mark.parametrize("dim1", [224])
def test_eval(dim1: int) -> None:
    model = ECGClassifier(lr=0.001, num_classes=3)
    x = torch.randn(1, 1, dim1, 224)  # ECG data: 1 channel
    y = model(x)
    assert y.shape[0] == 1, "Batch size should be 1"
    assert len(y.shape) == 2, "Output should be 2D"

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_model_with_different_batch_sizes(batch_size: int) -> None:
    model = ECGClassifier(lr=0.001, num_classes=3)
    x = torch.randn(batch_size, 1, 224, 224)  # ECG data: 1 channel, 224x224
    y = model(x)
    assert y.shape[0] == batch_size, f"Output batch size should be {batch_size}"
    assert len(y.shape) == 2, "Output should be 2D (batch_size, num_classes)"

