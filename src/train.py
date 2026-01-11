import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from src.data import ECGDataModule
from src.model import ECGClassifier


def main(args):
    seed_everything(42)

    # Data
    data_module = ECGDataModule(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model = ECGClassifier(
        lr=args.lr,
        num_classes=3
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="ecg-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    # Profiler
    profiler = PyTorchProfiler(
        dirpath="profiler",
        filename="perf_logs",
        export_to_chrome=True,
        row_limit=20,
        sort_by_key="cpu_time_total",
    )

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=TensorBoardLogger("lightning_logs", name="ecg_classification"),
        accelerator="auto",
        devices="auto",
        profiler=profiler,
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Classification Training")
    
    # Data params
    parser.add_argument("--data_dir", type=str, default="data/time_series", help="Path to raw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to save processed .pt files")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Model params
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Trainer params
    parser.add_argument("--max_epochs", type=int, default=2, help="Maximum number of epochs")

    args = parser.parse_args()
    
    main(args)
