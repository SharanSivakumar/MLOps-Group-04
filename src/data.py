from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class ECGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/time_series",
        processed_dir: str = "data/processed",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = ["AF", "Noise", "NSR"]  # Other is ignored

    def prepare_data(self):
        """
        Load .npy files, filter 'Other', split, and save as .pt tensors.
        This runs only on 1 GPU in distributed setting.
        """
        if (self.processed_dir / "train.pt").exists():
            print("Processed data found. Skipping preparation.")
            return

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        print("Preparing data...")
        X = []
        y = []

        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                print(f"Warning: Directory {cls_dir} does not exist. Skipping.")
                continue

            files = list(cls_dir.glob("*.npy"))
            print(f"Loading {len(files)} files from {cls_name}...")

            for file_path in files:
                # Load numpy file (224, 224) float64
                data = np.load(file_path).astype(np.float32)
                # Expand dims to (1, 224, 224)
                data = np.expand_dims(data, axis=0)
                X.append(data)
                y.append(class_to_idx[cls_name])

        if not X:
            raise RuntimeError("No data found! Check your data paths.")

        X = np.stack(X)  # (N, 1, 224, 224)
        y = np.array(y)

        # Stratified Split: 60% Train, 20% Val, 20% Test
        # First split: 80% Train+Val, 20% Test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Second split: From the 80%, we need 60% of total for Train, so 0.75 of 0.8 = 0.6
        # The remaining 0.25 of 0.8 = 0.2 which is val.
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )

        print(f"Saving splits: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

        torch.save({"x": torch.from_numpy(X_train), "y": torch.from_numpy(y_train)}, self.processed_dir / "train.pt")
        torch.save({"x": torch.from_numpy(X_val), "y": torch.from_numpy(y_val)}, self.processed_dir / "val.pt")
        torch.save({"x": torch.from_numpy(X_test), "y": torch.from_numpy(y_test)}, self.processed_dir / "test.pt")
        print("Data preparation complete.")

    def setup(self, stage: Optional[str] = None):
        """
        Load .pt files into memory.
        """
        if stage == "fit" or stage is None:
            train_data = torch.load(self.processed_dir / "train.pt")
            self.train_dataset = TensorDataset(train_data["x"], train_data["y"])

            val_data = torch.load(self.processed_dir / "val.pt")
            self.val_dataset = TensorDataset(val_data["x"], val_data["y"])

        if stage == "test" or stage is None:
            test_data = torch.load(self.processed_dir / "test.pt")
            self.test_dataset = TensorDataset(test_data["x"], test_data["y"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    # Test the data module
    dm = ECGDataModule()
    dm.prepare_data()
    dm.setup()
    print("DataModule setup successful.")
    for batch in dm.train_dataloader():
        x, y = batch
        print(f"Batch shape: {x.shape}, Label shape: {y.shape}")
        break
