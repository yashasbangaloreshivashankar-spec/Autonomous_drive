"""Configuration management for the Jetson Autopilot system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import json


@dataclass
class CameraConfig:
    width: int = 448
    height: int = 336
    fps: int = 10


@dataclass
class ModelConfig:
    frame_size: int = 224
    frame_channels: int = 3
    output_size: int = 2
    dropout_prob: float = 0.5
    backbone: Literal["resnet18", "resnet34", "mobilenet_v3"] = "resnet18"
    use_attention: bool = True
    use_uncertainty: bool = True
    use_multi_scale: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 128
    max_epochs: int = 50
    early_stopping_patience: int = 10
    initial_lr: float = 0.0005
    lr_reducer_patience: int = 2
    lr_reducer_factor: float = 0.9
    acceptable_testing_loss: float = 0.1
    use_one_cycle: bool = True
    use_mixup: bool = True
    use_amp: bool = True  # Automatic mixed precision
    use_ema: bool = True  # Exponential moving average


@dataclass
class AugmentationConfig:
    random_horizontal_flip: bool = True
    random_noise: bool = True
    random_blur: bool = True
    random_color_jitter: bool = True
    noise_amount: float = 0.1
    color_jitter_brightness: float = 0.25
    color_jitter_contrast: float = 0.25
    color_jitter_hue: float = 0.25
    color_jitter_saturation: float = 0.25
    # Advanced augmentations
    use_advanced: bool = True
    shadow_probability: float = 0.3
    perspective_probability: float = 0.2
    cutout_probability: float = 0.15
    motion_blur_probability: float = 0.1


@dataclass
class CarConfig:
    steering_offset: float = 0.035
    throttle_gain: float = 0.8


@dataclass
class InferenceConfig:
    use_tensorrt: bool = True
    use_smoothing: bool = True
    smoothing_method: Literal["ema", "median"] = "ema"
    smoothing_alpha: float = 0.4
    use_adaptive_throttle: bool = True
    adaptive_steering_sensitivity: float = 0.6
    adaptive_uncertainty_sensitivity: float = 2.0
    use_safety_monitor: bool = True
    safety_uncertainty_threshold: float = 0.25
    safety_max_steering_rate: float = 0.5


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    car: CarConfig = field(default_factory=CarConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    models_dir: Path = Path("models")
    datasets_dir: Path = Path("datasets")
    model_name: str = "autopilot"
    show_logs: bool = False

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.model_name}.pth"

    @property
    def model_trt_path(self) -> Path:
        return self.models_dir / f"{self.model_name}_trt.pth"

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        with open(path) as f:
            data = json.load(f)

        config = cls()

        if "camera" in data:
            config.camera = CameraConfig(**data["camera"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "augmentation" in data:
            config.augmentation = AugmentationConfig(**data["augmentation"])
        if "car" in data:
            config.car = CarConfig(**data["car"])
        if "inference" in data:
            config.inference = InferenceConfig(**data["inference"])

        for key in ["models_dir", "datasets_dir"]:
            if key in data:
                setattr(config, key, Path(data[key]))

        for key in ["model_name", "show_logs"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_json(self, path: Path) -> None:
        data = {
            "camera": self.camera.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "augmentation": self.augmentation.__dict__,
            "car": self.car.__dict__,
            "inference": self.inference.__dict__,
            "models_dir": str(self.models_dir),
            "datasets_dir": str(self.datasets_dir),
            "model_name": self.model_name,
            "show_logs": self.show_logs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def get_default_config() -> Config:
    return Config()


# Preset configurations for different use cases
def get_fast_config() -> Config:
    """Configuration optimized for speed (lower accuracy)."""
    config = Config()
    config.model.backbone = "mobilenet_v3"
    config.model.use_multi_scale = False
    config.model.use_uncertainty = False
    config.inference.use_smoothing = True
    config.inference.use_safety_monitor = False
    return config


def get_accurate_config() -> Config:
    """Configuration optimized for accuracy (slower)."""
    config = Config()
    config.model.backbone = "resnet34"
    config.model.use_attention = True
    config.model.use_uncertainty = True
    config.model.use_multi_scale = True
    config.training.max_epochs = 100
    config.training.batch_size = 64
    return config


def get_safe_config() -> Config:
    """Configuration optimized for safety."""
    config = Config()
    config.model.use_uncertainty = True
    config.inference.use_safety_monitor = True
    config.inference.safety_uncertainty_threshold = 0.15
    config.inference.use_adaptive_throttle = True
    config.inference.adaptive_uncertainty_sensitivity = 3.0
    config.car.throttle_gain = 0.6  # More conservative
    return config
