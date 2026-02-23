"""Jetson Autopilot - Smart self-driving toy car system.

An advanced CNN-based autopilot system for NVIDIA Jetson with:
- Attention mechanisms (CBAM) for focusing on road features
- Uncertainty estimation for safety-aware driving
- Multi-scale feature fusion for better perception
- Temporal smoothing for stable control
- Adaptive throttle control
- Safety monitoring with emergency stops

Example usage:

    # Training
    from jetson import Config, AutopilotModel, Trainer, create_data_loaders

    config = Config()
    train_loader, val_loader, _ = create_data_loaders(
        config, training_dir="data/train", validation_dir="data/val"
    )
    model = AutopilotModel(config=config.model)
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)

    # Inference
    from jetson import AutopilotController

    controller = AutopilotController(config)
    controller.setup(model_path="models/autopilot.pth")
    controller.run()
"""

from .config import (
    Config,
    CameraConfig,
    ModelConfig,
    TrainingConfig,
    AugmentationConfig,
    CarConfig,
    InferenceConfig,
    get_default_config,
    get_fast_config,
    get_accurate_config,
    get_safe_config,
)
from .model import (
    AutopilotModel,
    SpatialAttention,
    ChannelAttention,
    CBAM,
    MultiScaleFeatureFusion,
    UncertaintyHead,
    GaussianNLLLoss,
)
from .dataset import (
    AutopilotDataset,
    MixupDataset,
    AdvancedAugmentations,
    create_data_loaders,
)
from .preprocessing import (
    center_crop_square,
    ImagePreprocessor,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .trainer import (
    Trainer,
    EarlyStopping,
    GradientClipping,
)
from .inference import (
    InferenceEngine,
    AutopilotController,
    TemporalSmoother,
    AdaptiveThrottleController,
    SafetyMonitor,
)
from .data_collection import DataCollector

__version__ = "2.0.0"
__author__ = "Greg Surma"

__all__ = [
    # Config
    "Config",
    "CameraConfig",
    "ModelConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "CarConfig",
    "InferenceConfig",
    "get_default_config",
    "get_fast_config",
    "get_accurate_config",
    "get_safe_config",
    # Model
    "AutopilotModel",
    "SpatialAttention",
    "ChannelAttention",
    "CBAM",
    "MultiScaleFeatureFusion",
    "UncertaintyHead",
    "GaussianNLLLoss",
    # Dataset
    "AutopilotDataset",
    "MixupDataset",
    "AdvancedAugmentations",
    "create_data_loaders",
    # Preprocessing
    "center_crop_square",
    "ImagePreprocessor",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # Training
    "Trainer",
    "EarlyStopping",
    "GradientClipping",
    # Inference
    "InferenceEngine",
    "AutopilotController",
    "TemporalSmoother",
    "AdaptiveThrottleController",
    "SafetyMonitor",
    # Data Collection
    "DataCollector",
]
