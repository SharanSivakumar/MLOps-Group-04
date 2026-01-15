import pytest
import torch
from pathlib import Path

from src.data import ECGDataModule


def test_data_module_initialization():
    """Test that ECGDataModule initializes with correct parameters."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    
    assert data_module.batch_size == 32, "Batch size should be 32"
    assert data_module.num_workers == 0, "Num workers should be 0"
    assert data_module.classes == ["AF", "Noise", "NSR"], "Should have 3 classes"


def test_data_module_with_different_batch_sizes():
    """Test that ECGDataModule works with different batch sizes."""
    for batch_size in [8, 16, 32, 64]:
        data_module = ECGDataModule(
            data_dir="data/raw",
            processed_dir="data/processed",
            batch_size=batch_size,
            num_workers=0
        )
        assert data_module.batch_size == batch_size, f"Batch size should be {batch_size}"


def test_data():
    """Test basic data loading and shape validation."""
    # Initialize data module with test parameters
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.prepare_data()
    data_module.setup()
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    assert len(train_loader) > 0, "training set should contain batches"
    assert len(val_loader) > 0, "validation set should contain batches"
    assert len(test_loader) > 0, "test set should contain batches"
    
    # Check a batch from train loader
    for x, y in train_loader:
        assert x.shape[1] == 1, "ECG data should have 1 channel"
        assert x.shape[2] == 224, "input height should be 224"
        assert x.shape[3] == 224, "input width should be 224"
        assert isinstance(y, torch.Tensor), "target label should be a tensor"
        assert y.dtype in [torch.long, torch.int64, torch.int32], "target label should be an integer type"
        break  # Only check first batch


def test_dataloader_shapes():
    """Test that all dataloaders return correct tensor shapes."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=16,
        num_workers=0
    )
    data_module.setup()
    
    # Test train loader
    train_loader = data_module.train_dataloader()
    for x, y in train_loader:
        assert len(x.shape) == 4, "Input should be 4D tensor (batch, channel, height, width)"
        assert x.shape[1] == 1, "Should have 1 channel"
        assert x.shape[2] == 224, "Height should be 224"
        assert x.shape[3] == 224, "Width should be 224"
        assert len(y.shape) == 1, "Labels should be 1D tensor"
        assert x.shape[0] == y.shape[0], "Batch sizes should match"
        break
    
    # Test validation loader
    val_loader = data_module.val_dataloader()
    for x, y in val_loader:
        assert x.shape[1:] == (1, 224, 224), "Validation data shape should match training"
        break
    
    # Test test loader
    test_loader = data_module.test_dataloader()
    for x, y in test_loader:
        assert x.shape[1:] == (1, 224, 224), "Test data shape should match training"
        break


def test_data_splits():
    """Test that data is split correctly into train/val/test sets."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.setup()
    
    train_size = len(data_module.train_dataset)
    val_size = len(data_module.val_dataset)
    test_size = len(data_module.test_dataset)
    total_size = train_size + val_size + test_size
    
    # Check that splits are approximately 60/20/20
    train_ratio = train_size / total_size
    val_ratio = val_size / total_size
    test_ratio = test_size / total_size
    
    assert 0.55 < train_ratio < 0.65, f"Train split should be ~60%, got {train_ratio:.2%}"
    assert 0.15 < val_ratio < 0.25, f"Val split should be ~20%, got {val_ratio:.2%}"
    assert 0.15 < test_ratio < 0.25, f"Test split should be ~20%, got {test_ratio:.2%}"


def test_class_labels():
    """Test that class labels are in the correct range."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.setup()
    
    # Check all loaders
    for loader_name, loader in [
        ("train", data_module.train_dataloader()),
        ("val", data_module.val_dataloader()),
        ("test", data_module.test_dataloader())
    ]:
        for _, y in loader:
            assert (y >= 0).all(), f"{loader_name} labels should be >= 0"
            assert (y < 3).all(), f"{loader_name} labels should be < 3 (3 classes)"
            break


def test_data_types():
    """Test that data has correct dtypes."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    for x, y in train_loader:
        assert x.dtype == torch.float32, "Input data should be float32"
        assert y.dtype in [torch.long, torch.int64], "Labels should be long/int64"
        break


def test_data_values_are_valid():
    """Test that data doesn't contain NaN or Inf values."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    for x, y in train_loader:
        assert not torch.isnan(x).any(), "Data should not contain NaN values"
        assert not torch.isinf(x).any(), "Data should not contain Inf values"
        assert torch.isfinite(x).all(), "All data values should be finite"
        break


def test_train_loader_shuffles():
    """Test that training loader uses random sampling (shuffle=True)."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    # Check that sampler is RandomSampler (used when shuffle=True)
    from torch.utils.data import RandomSampler
    assert isinstance(train_loader.sampler, RandomSampler), "Training loader should use RandomSampler (shuffle=True)"


def test_val_test_loaders_dont_shuffle():
    """Test that validation and test loaders don't shuffle."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.setup()
    
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Check that samplers are SequentialSampler (used when shuffle=False)
    from torch.utils.data import SequentialSampler
    assert isinstance(val_loader.sampler, SequentialSampler), "Validation loader should use SequentialSampler (shuffle=False)"
    assert isinstance(test_loader.sampler, SequentialSampler), "Test loader should use SequentialSampler (shuffle=False)"


@pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
def test_different_batch_sizes(batch_size):
    """Test that dataloaders work with different batch sizes."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=batch_size,
        num_workers=0
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    for x, y in train_loader:
        # Last batch might be smaller, but should be <= batch_size
        assert x.shape[0] <= batch_size, f"Batch size should be <= {batch_size}"
        assert x.shape[0] == y.shape[0], "X and y batch sizes should match"
        break


def test_datasets_exist_after_setup():
    """Test that datasets are created after setup."""
    data_module = ECGDataModule(
        data_dir="data/raw",
        processed_dir="data/processed",
        batch_size=32,
        num_workers=0
    )
    data_module.setup()
    
    assert hasattr(data_module, 'train_dataset'), "Should have train_dataset"
    assert hasattr(data_module, 'val_dataset'), "Should have val_dataset"
    assert hasattr(data_module, 'test_dataset'), "Should have test_dataset"
    
    assert data_module.train_dataset is not None, "train_dataset should not be None"
    assert data_module.val_dataset is not None, "val_dataset should not be None"
    assert data_module.test_dataset is not None, "test_dataset should not be None"

