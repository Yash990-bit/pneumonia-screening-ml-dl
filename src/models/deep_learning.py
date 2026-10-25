"""Deep learning architectures and transfer-learning module for Pneumonia Screening.

Implements Shitanshu's responsibilities:
- ResNet18 transfer learning (ImageNet pretrained, frozen feature extractor, fine-tuning)
- MobileNetV2 transfer learning (lightweight mobile architecture)
- CNN embedding extraction for downstream hybrid feature fusion
- Hybrid classification (HOG/LBP + CNN embeddings)

Educational research prototype — NOT intended for clinical diagnosis.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

BackboneName = Literal["resnet18", "mobilenet_v2"]


class ChestXrayClassifier(nn.Module):
    """Transfer-learned CNN for Chest X-Ray pneumonia classification."""

    def __init__(
        self,
        backbone_name: BackboneName = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_p: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)

            # Freeze or unfreeze feature extractor layers
            if freeze_backbone:
                for param in base.parameters():
                    param.requires_grad = False

            # Extract embedding dimension
            in_features = base.fc.in_features  # 512 for ResNet18
            self.embedding_dim = in_features

            # Separate feature extractor and pooling
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
            )
            self.avgpool = base.avgpool
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )

        elif backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)

            if freeze_backbone:
                for param in base.parameters():
                    param.requires_grad = False

            in_features = base.last_channel  # 1280 for MobileNetV2
            self.embedding_dim = in_features

            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the final convolutional layer for Grad-CAM."""
        if self.backbone_name == "resnet18":
            return self.features[-1][-1].conv2
        return self.features[-1]

    def unfreeze_late_layers(self, num_blocks: int = 1) -> None:
        """Unfreeze late convolutional layers for fine-tuning with a smaller learning rate."""
        if self.backbone_name == "resnet18":
            # layer4 is features[7]
            for param in self.features[7].parameters():
                param.requires_grad = True
        elif self.backbone_name == "mobilenet_v2":
            # unfreeze the last few blocks of MobileNetV2 features
            for layer in self.features[-3:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embedding vector."""
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)
        return flattened

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        embedding = self.extract_embedding(x)
        logits = self.head(embedding)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted pneumonia probability via sigmoid."""
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)[:, 1:2]
