from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from src.data import ECGDataModule
from src.model import ECGClassifier


def main(config):
    seed_everything(config.seed)

    # Data
    data_module = ECGDataModule(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = ECGClassifier(lr=args.lr, num_classes=3)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        dirpath=config.callbacks.checkpoint.dirpath,
        filename=config.callbacks.checkpoint.filename,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        mode=config.callbacks.checkpoint.mode,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Profiler
    profiler = PyTorchProfiler(
        dirpath=config.profiler.dirpath,
        filename=config.profiler.filename,
        export_to_chrome=config.profiler.export_to_chrome,
        row_limit=config.profiler.row_limit,
        sort_by_key=config.profiler.sort_by_key,
    )

    # Trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=TensorBoardLogger(config.logging.log_dir, name=config.logging.name),
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
