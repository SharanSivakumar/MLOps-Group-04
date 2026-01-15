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
        data_dir=config.data.data_dir,
        processed_dir=config.data.processed_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )

    # Model
    model = ECGClassifier(
        lr=config.model.lr,
        num_classes=config.model.num_classes
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        dirpath=config.callbacks.checkpoint.dirpath,
        filename=config.callbacks.checkpoint.filename,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        mode=config.callbacks.checkpoint.mode,
    )
    early_stopping = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode
    )
    
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
    # Load configuration from YAML file
    config = OmegaConf.load("config.yaml")
    
    main(config)
