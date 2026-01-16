import torch
from src.model import ECGClassifier


def test_model_architecture():
    model = ECGClassifier()
    batch_size = 1
    channels = 1
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, channels, height, width)

    output = model(dummy_input)

    expected_output_shape = (batch_size, 3)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"
