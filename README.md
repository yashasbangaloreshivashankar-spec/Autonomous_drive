# Autonomous_drive
# Jetson Autopilot v2.0

A **smart** self-driving toy car system using advanced CNN-based computer vision for NVIDIA Jetson.

## What's New in v2.0

| Feature | Description |
|---------|-------------|
| **CBAM Attention** | Model focuses on road-relevant features |
| **Uncertainty Estimation** | Know when the model is unsure |
| **Multi-Scale Fusion** | Better perception at different distances |
| **Adaptive Throttle** | Automatically slows on sharp turns |
| **Temporal Smoothing** | No more jittery steering |
| **Safety Monitor** | Emergency stops on high uncertainty |
| **Advanced Augmentations** | Shadows, perspective, cutout, motion blur |
| **Mixed Precision Training** | 2x faster training with AMP |
| **Mixup & EMA** | Better generalization |

## Architecture

```
Camera Frame
     │
     ▼
┌─────────────────────────────────────┐
│  ResNet18/34 Backbone               │
│  + Multi-Scale Feature Fusion       │
│  + CBAM Attention                   │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Prediction Head                    │
│  + Uncertainty Estimation           │
│  → [steering, throttle, σ]          │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Smart Inference                    │
│  • Temporal Smoothing (EMA)         │
│  • Adaptive Throttle                │
│  • Safety Monitor                   │
└─────────────────────────────────────┘
     │
     ▼
  Car Control
```

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Train with Smart Features

```bash
python -m jetson train \
    --training-dir datasets/training \
    --validation-dir datasets/validation \
    --model-name smart_autopilot
```

### 2. Run with Safety Features

```bash
python -m jetson run \
    --model-path models/smart_autopilot.pth \
    --show-fps
```

## Configuration Presets

```python
from jetson import get_default_config, get_fast_config, get_safe_config, get_accurate_config

# Balanced (default)
config = get_default_config()

# Speed optimized (~45 FPS)
config = get_fast_config()  # MobileNetV3, no uncertainty

# Safety optimized
config = get_safe_config()  # Lower throttle, strict monitoring

# Accuracy optimized
config = get_accurate_config()  # ResNet34, all features
```

## Smart Features Explained

### 1. CBAM Attention
The model learns to focus on important regions (road edges, lane markings) and ignore distractions (sky, trees).

```python
# Automatically enabled
model = AutopilotModel(use_attention=True)
```

### 2. Uncertainty Estimation
The model outputs confidence along with predictions. High uncertainty = slow down or stop.

```python
# Get prediction with uncertainty
steering, throttle, metadata = engine.predict(frame)
print(f"Uncertainty: {metadata['uncertainty']:.3f}")
```

### 3. Adaptive Throttle
Automatically reduces speed based on:
- **Steering angle**: Sharp turn → slow down
- **Uncertainty**: Unsure → slow down

```python
# Configured via
config.inference.adaptive_steering_sensitivity = 0.6
config.inference.adaptive_uncertainty_sensitivity = 2.0
```

### 4. Temporal Smoothing
Exponential moving average prevents jerky movements:

```python
config.inference.smoothing_method = "ema"
config.inference.smoothing_alpha = 0.4  # Higher = more responsive
```

### 5. Safety Monitor
Triggers warnings/emergency stops when:
- Uncertainty exceeds threshold for multiple frames
- Steering changes too rapidly

```python
config.inference.safety_uncertainty_threshold = 0.25
config.inference.safety_max_steering_rate = 0.5
```

## Advanced Augmentations

Training includes realistic augmentations:

| Augmentation | Effect |
|--------------|--------|
| Shadow | Simulates varying lighting |
| Perspective | Different viewing angles |
| Cutout | Occlusion robustness |
| Motion Blur | Fast movement simulation |
| Brightness Gradient | Sun glare effects |

## Training Techniques

- **AdamW** with weight decay for better generalization
- **OneCycle LR** with warmup for faster convergence
- **Mixed Precision (AMP)** for 2x training speed
- **Exponential Moving Average** for smoother final model
- **Mixup** for better interpolation between samples
- **Gradient Clipping** for training stability

## API Examples

### Custom Training Loop

```python
from jetson import Config, AutopilotModel, Trainer, create_data_loaders

config = Config()
config.model.backbone = "resnet34"
config.training.use_one_cycle = True
config.training.use_amp = True

train_loader, val_loader, test_loader = create_data_loaders(
    config,
    training_dir="data/train",
    validation_dir="data/val",
    testing_dir="data/test",
    use_mixup=True,
)

model = AutopilotModel(
    config=config.model,
    pretrained=True,
    use_attention=True,
    use_uncertainty=True,
)

trainer = Trainer(model, config, use_amp=True, use_ema=True)
history = trainer.train(train_loader, val_loader)

# Test with uncertainty
avg_loss, results = trainer.test(test_loader, use_uncertainty=True)
for r in results:
    print(f"{r['name']}: loss={r['loss']:.4f}, uncertainty={r['uncertainty']}")
```

### Custom Inference Pipeline

```python
from jetson import InferenceEngine, TemporalSmoother, SafetyMonitor

engine = InferenceEngine(
    use_tensorrt=True,
    use_smoothing=True,
    use_adaptive_throttle=True,
    use_safety_monitor=True,
)
engine.load_model("models/autopilot.pth")

# Single prediction
steering, throttle, metadata = engine.predict(frame)

if not metadata['safe']:
    print(f"Warning: {metadata['reason']}")
    # Take corrective action
```

## Project Structure

```
jetson/
├── config.py          # Dataclass configs with presets
├── model.py           # CNN with attention, uncertainty, multi-scale
├── dataset.py         # Advanced augmentations, mixup
├── preprocessing.py   # Image preprocessing
├── trainer.py         # AMP, EMA, OneCycle, gradient clipping
├── inference.py       # Smoothing, adaptive throttle, safety
├── data_collection.py # Data recording utilities
└── cli.py             # Command-line interface
```

## Performance

| Configuration | FPS | Features |
|--------------|-----|----------|
| Fast (MobileNetV3) | ~45 | Basic |
| Default (ResNet18) | ~30 | Full |
| Accurate (ResNet34) | ~20 | Full + larger model |

## Hardware Requirements

- **NVIDIA Jetson Nano** or better
- RC car with controllable steering/throttle
- Wide-angle camera (200° FOV recommended)

## Dependencies

Core:
- PyTorch >= 1.9
- torchvision >= 0.10
- OpenCV >= 4.5
- NumPy, Pillow, tqdm

Jetson-specific:
- jetcam, jetracer
- torch2trt (for TensorRT optimization)

## Author

**Yashas**
##
