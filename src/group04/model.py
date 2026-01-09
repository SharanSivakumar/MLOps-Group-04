from torch import nn
import torch
from pytorch_lightning import LightningModule



class Model(LightningModule):
    # import EfficientNet B4 model as classifier for 3 categories
    def __init__(self):
        super().__init__()
        self.efficientnet = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b4",
            pretrained=True,
        )
        self.classifier = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
