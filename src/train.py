import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from loguru import logger
import wandb

from src.data import ECGDataModule
from src.model import ECGClassifier

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    seed_everything(config.seed)

    logger.info("Starting ECG classification training")
    logger.info(f"Configuration: batch_size={config.data.batch_size}, lr={config.model.lr}, max_epochs={config.training.max_epochs}, seed={config.seed}")

    # Data
    data_module = ECGDataModule(
        data_dir=config.data.data_dir,
        processed_dir=config.data.processed_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )
    logger.info(f"Data module initialized from {config.data.data_dir}")

    # Model
    model = ECGClassifier(
        lr=config.model.lr,
        num_classes=config.model.num_classes
    )
    logger.info(f"Model initialized with lr={config.model.lr}, num_classes={config.model.num_classes}")

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

    # Weights & Biases Logger
    wandb_logger = WandbLogger(
        project=config.logging.get('project', 'ecg-classification'),
        name=config.logging.get('name', 'ecg-experiment'),
        config={
            'batch_size': config.data.batch_size,
            'lr': config.model.lr,
            'max_epochs': config.training.max_epochs,
            'seed': config.seed,
            'num_classes': config.model.num_classes
        }
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        profiler=profiler,
    )

    logger.info(f"Starting training for {config.training.max_epochs} epochs")
    # Train
    trainer.fit(model, data_module)
    logger.success("Training completed successfully")

    logger.info("Starting model evaluation on test set")
    # Test
    test_results = trainer.test(model, data_module)
    logger.info("Testing completed")
    
    # Log model as W&B artifact
    logger.info("Logging model as W&B artifact")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        artifact = wandb.Artifact(
            name=f"{config.logging.get('name', 'ecg-model')}",
            type="model",
            description="ECG classification model trained with PyTorch Lightning",
            metadata={
                'test_results': test_results[0] if test_results else {},
                'best_checkpoint': best_model_path
            }
        )
        artifact.add_file(best_model_path)
        wandb_logger.experiment.log_artifact(artifact)
        logger.success(f"Model artifact logged: {best_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
