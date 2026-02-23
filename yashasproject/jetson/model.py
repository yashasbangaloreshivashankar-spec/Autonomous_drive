"""Neural network model for autopilot steering and throttle prediction."""

from typing import Optional, Tuple
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .config import ModelConfig, get_default_config


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important image regions."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation style)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiScaleFeatureFusion(nn.Module):
    """Fuse features from multiple scales for better perception."""

    def __init__(self, in_channels_list: list, out_channels: int = 256) -> None:
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, features: list) -> torch.Tensor:
        target_size = features[-1].shape[2:]

        fused = []
        for feat, lateral in zip(features, self.lateral_convs):
            out = lateral(feat)
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            fused.append(out)

        fused = torch.cat(fused, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        return F.relu(fused, inplace=True)


class UncertaintyHead(nn.Module):
    """Predicts both values and uncertainty (aleatoric uncertainty)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.mean_head = nn.Linear(in_features, out_features)
        self.log_var_head = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        # Clamp log variance for stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        return mean, log_var


class AutopilotModel(nn.Module):
    """Advanced CNN model with attention and uncertainty estimation.

    Features:
    - ResNet backbone with multi-scale feature fusion
    - CBAM attention for focusing on road features
    - Uncertainty estimation for safety
    - Dropout for regularization and MC-Dropout inference
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        pretrained: bool = True,
        use_attention: bool = True,
        use_uncertainty: bool = True,
        use_multi_scale: bool = True,
    ) -> None:
        super().__init__()

        if config is None:
            config = get_default_config().model

        self.config = config
        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty
        self.use_multi_scale = use_multi_scale

        self._build_network(pretrained)

    def _build_network(self, pretrained: bool) -> None:
        # Load backbone
        if self.config.backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            layer_channels = [64, 128, 256, 512]
        elif self.config.backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            layer_channels = [64, 128, 256, 512]
        elif self.config.backbone == "mobilenet_v3":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            # MobileNetV3 has different structure, use simpler approach
            self.use_multi_scale = False
            layer_channels = [576]  # Last feature map channels
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone}")

        # Extract backbone layers
        if "resnet" in self.config.backbone:
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.avgpool = backbone.avgpool
            feature_dim = 512
        else:
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            feature_dim = 576

        # Multi-scale feature fusion
        if self.use_multi_scale and "resnet" in self.config.backbone:
            self.multi_scale = MultiScaleFeatureFusion(layer_channels, out_channels=256)
            self.ms_pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 256

        # Attention module
        if self.use_attention:
            self.attention = CBAM(feature_dim if not self.use_multi_scale else 512)

        # Prediction head
        hidden_dim = 256
        self.head = nn.Sequential(
            nn.Dropout(p=self.config.dropout_prob),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=self.config.dropout_prob),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=self.config.dropout_prob / 2),
        )

        # Output heads
        if self.use_uncertainty:
            self.output = UncertaintyHead(128, self.config.output_size)
        else:
            self.output = nn.Linear(128, self.config.output_size)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'features'):
            # MobileNet path
            x = self.features(x)
            if self.use_attention:
                x = self.attention(x)
            x = self.avgpool(x)
            return x.flatten(1)

        # ResNet path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        if self.use_attention:
            f4 = self.attention(f4)

        if self.use_multi_scale:
            fused = self.multi_scale([f1, f2, f3, f4])
            x = self.ms_pool(fused)
        else:
            x = self.avgpool(f4)

        return x.flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self._forward_backbone(x)
        features = self.head(features)

        if self.use_uncertainty:
            mean, log_var = self.output(features)
            if return_uncertainty:
                uncertainty = torch.exp(0.5 * log_var)  # Standard deviation
                return mean, uncertainty
            return mean
        else:
            return self.output(features)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout for epistemic uncertainty estimation.

        Args:
            x: Input tensor.
            n_samples: Number of forward passes.

        Returns:
            Tuple of (mean_prediction, aleatoric_uncertainty, epistemic_uncertainty)
        """
        self.train()  # Enable dropout

        predictions = []
        aleatoric_vars = []

        with torch.no_grad():
            for _ in range(n_samples):
                if self.use_uncertainty:
                    mean, log_var = self.output(self.head(self._forward_backbone(x)))
                    predictions.append(mean)
                    aleatoric_vars.append(torch.exp(log_var))
                else:
                    pred = self.forward(x)
                    predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        epistemic_var = predictions.var(dim=0)

        if self.use_uncertainty:
            aleatoric_var = torch.stack(aleatoric_vars).mean(dim=0)
            total_uncertainty = torch.sqrt(aleatoric_var + epistemic_var)
        else:
            total_uncertainty = torch.sqrt(epistemic_var)

        self.eval()
        return mean_pred, torch.sqrt(aleatoric_var) if self.use_uncertainty else None, torch.sqrt(epistemic_var)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'use_attention': self.use_attention,
            'use_uncertainty': self.use_uncertainty,
            'use_multi_scale': self.use_multi_scale,
        }, path)

    def load(self, path: Path, device: Optional[torch.device] = None) -> None:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self.load_state_dict(checkpoint)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: Optional[torch.device] = None,
    ) -> "AutopilotModel":
        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            model = cls(
                config=checkpoint['config'],
                pretrained=False,
                use_attention=checkpoint.get('use_attention', True),
                use_uncertainty=checkpoint.get('use_uncertainty', True),
                use_multi_scale=checkpoint.get('use_multi_scale', True),
            )
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = cls(pretrained=False)
            model.load_state_dict(checkpoint)

        return model


class GaussianNLLLoss(nn.Module):
    """Negative Log-Likelihood loss for uncertainty estimation."""

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        precision = torch.exp(-log_var)
        loss = 0.5 * (precision * (target - mean) ** 2 + log_var)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
