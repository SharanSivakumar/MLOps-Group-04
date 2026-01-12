import os
import pytest
from src.data import ECGDataModule


@pytest.mark.skipif(not os.path.exists("data/time_series"), reason="Data files not found")
def test_ecg_data_module():
    data_dir = "data/time_series"
    dm = ECGDataModule(data_dir=data_dir, processed_dir="data/processed", batch_size=4)

    dm.prepare_data()
    dm.setup()

    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")
    assert hasattr(dm, "test_dataset")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    x, y = batch

    assert x.shape == (4, 1, 224, 224)
    assert y.shape == (4,)
