import os
from pathlib import Path

import pytest
import torch

from src.model import ECGClassifier
from src.train import main

# Testing the training can take a long time, so wait till later
def test_training_run(tmp_path: Path) -> None:
    pass


