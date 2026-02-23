"""Smart real-time inference with safety features."""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Deque
from collections import deque
import math

import numpy as np
import torch

from .config import Config, get_default_config
from .model import AutopilotModel
from .preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Smooth predictions over time to reduce jitter."""

    def __init__(
        self,
        window_size: int = 5,
        method: str = "ema",  # "ema" or "median"
        ema_alpha: float = 0.3,
    ) -> None:
        self.window_size = window_size
        self.method = method
        self.ema_alpha = ema_alpha

        self.steering_history: Deque[float] = deque(maxlen=window_size)
        self.throttle_history: Deque[float] = deque(maxlen=window_size)

        self._ema_steering = 0.0
        self._ema_throttle = 0.0
        self._initialized = False

    def smooth(self, steering: float, throttle: float) -> Tuple[float, float]:
        if self.method == "ema":
            return self._ema_smooth(steering, throttle)
        else:
            return self._median_smooth(steering, throttle)

    def _ema_smooth(self, steering: float, throttle: float) -> Tuple[float, float]:
        if not self._initialized:
            self._ema_steering = steering
            self._ema_throttle = throttle
            self._initialized = True
        else:
            self._ema_steering = (
                self.ema_alpha * steering + (1 - self.ema_alpha) * self._ema_steering
            )
            self._ema_throttle = (
                self.ema_alpha * throttle + (1 - self.ema_alpha) * self._ema_throttle
            )

        return self._ema_steering, self._ema_throttle

    def _median_smooth(self, steering: float, throttle: float) -> Tuple[float, float]:
        self.steering_history.append(steering)
        self.throttle_history.append(throttle)

        return (
            float(np.median(list(self.steering_history))),
            float(np.median(list(self.throttle_history))),
        )

    def reset(self) -> None:
        self.steering_history.clear()
        self.throttle_history.clear()
        self._initialized = False


class AdaptiveThrottleController:
    """Adjusts throttle based on steering and uncertainty."""

    def __init__(
        self,
        base_throttle: float = 0.5,
        min_throttle: float = 0.2,
        max_throttle: float = 0.8,
        steering_sensitivity: float = 0.5,
        uncertainty_sensitivity: float = 2.0,
    ) -> None:
        self.base_throttle = base_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.steering_sensitivity = steering_sensitivity
        self.uncertainty_sensitivity = uncertainty_sensitivity

    def compute_throttle(
        self,
        predicted_throttle: float,
        steering: float,
        uncertainty: Optional[float] = None,
    ) -> float:
        """Compute adaptive throttle based on steering and uncertainty.

        - Slows down on sharp turns
        - Slows down when model is uncertain
        """
        # Base throttle from model
        throttle = predicted_throttle

        # Reduce throttle on sharp turns (quadratic reduction)
        steering_factor = 1.0 - self.steering_sensitivity * (steering ** 2)
        throttle *= max(0.3, steering_factor)

        # Reduce throttle when uncertain
        if uncertainty is not None:
            uncertainty_factor = 1.0 / (1.0 + self.uncertainty_sensitivity * uncertainty)
            throttle *= uncertainty_factor

        # Clamp to valid range
        return float(np.clip(throttle, self.min_throttle, self.max_throttle))


class SafetyMonitor:
    """Monitors for unsafe conditions and triggers emergency stops."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        consecutive_uncertain_frames: int = 5,
        max_steering_rate: float = 0.5,  # Maximum steering change per frame
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.consecutive_uncertain_frames = consecutive_uncertain_frames
        self.max_steering_rate = max_steering_rate

        self._uncertain_count = 0
        self._last_steering = 0.0
        self._emergency_stop = False

    def check(
        self,
        steering: float,
        uncertainty: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Check if conditions are safe.

        Returns:
            Tuple of (is_safe, reason).
        """
        reasons = []

        # Check uncertainty
        if uncertainty is not None and uncertainty > self.uncertainty_threshold:
            self._uncertain_count += 1
            if self._uncertain_count >= self.consecutive_uncertain_frames:
                reasons.append(f"High uncertainty ({uncertainty:.3f}) for {self._uncertain_count} frames")
        else:
            self._uncertain_count = 0

        # Check steering rate
        steering_rate = abs(steering - self._last_steering)
        if steering_rate > self.max_steering_rate:
            reasons.append(f"Steering change too fast ({steering_rate:.3f})")

        self._last_steering = steering

        is_safe = len(reasons) == 0
        return is_safe, "; ".join(reasons) if reasons else "OK"

    def reset(self) -> None:
        self._uncertain_count = 0
        self._last_steering = 0.0
        self._emergency_stop = False


class InferenceEngine:
    """Smart inference engine with smoothing, adaptation, and safety."""

    def __init__(
        self,
        config: Optional[Config] = None,
        use_tensorrt: bool = True,
        use_smoothing: bool = True,
        use_adaptive_throttle: bool = True,
        use_safety_monitor: bool = True,
    ) -> None:
        self.config = config or get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tensorrt = use_tensorrt and self.device.type == "cuda"

        self.preprocessor = ImagePreprocessor(
            device=self.device,
            frame_size=self.config.model.frame_size,
        )

        self.model = None
        self._trt_available = False
        self._model_has_uncertainty = False

        # Smart components
        self.smoother = TemporalSmoother(
            window_size=5, method="ema", ema_alpha=0.4
        ) if use_smoothing else None

        self.throttle_controller = AdaptiveThrottleController(
            base_throttle=0.5,
            steering_sensitivity=0.6,
            uncertainty_sensitivity=2.0,
        ) if use_adaptive_throttle else None

        self.safety_monitor = SafetyMonitor(
            uncertainty_threshold=0.25,
            consecutive_uncertain_frames=5,
        ) if use_safety_monitor else None

        try:
            from torch2trt import TRTModule, torch2trt
            self._trt_available = True
        except ImportError:
            if self.use_tensorrt:
                logger.warning("torch2trt not available, falling back to standard PyTorch")
                self.use_tensorrt = False

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load the model."""

        if model_path is None:
            model_path = self.config.model_path

        trt_path = model_path.with_suffix("").with_suffix(".trt.pth")

        if self.use_tensorrt and self._trt_available:
            if trt_path.exists():
                logger.info(f"Loading TensorRT model from {trt_path}")
                self._load_trt_model(trt_path)
            else:
                logger.info(f"Converting model to TensorRT: {model_path} -> {trt_path}")
                self._convert_to_trt(model_path, trt_path)
        else:
            logger.info(f"Loading PyTorch model from {model_path}")
            self._load_pytorch_model(model_path)

    def _load_pytorch_model(self, path: Path) -> None:
        self.model = AutopilotModel.from_checkpoint(path, self.device)
        self.model.to(self.device)
        self.model.eval()
        self._model_has_uncertainty = self.model.use_uncertainty

    def _load_trt_model(self, path: Path) -> None:
        from torch2trt import TRTModule
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(path))
        self._model_has_uncertainty = False  # TRT doesn't support uncertainty

    def _convert_to_trt(self, model_path: Path, trt_path: Path) -> None:
        from torch2trt import torch2trt

        self._load_pytorch_model(model_path)

        # Create a simpler model for TRT conversion (no uncertainty)
        simple_model = AutopilotModel(
            config=self.config.model,
            pretrained=False,
            use_attention=True,
            use_uncertainty=False,
            use_multi_scale=True,
        )
        simple_model.load_state_dict(self.model.state_dict(), strict=False)
        simple_model.to(self.device)
        simple_model.eval()

        cfg = self.config.model
        dummy_input = torch.ones(
            (1, cfg.frame_channels, cfg.frame_size, cfg.frame_size),
            device=self.device,
        )

        self.model = torch2trt(simple_model, [dummy_input], fp16_mode=True)

        trt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), trt_path)

        logger.info(f"TensorRT model saved to {trt_path}")
        self._model_has_uncertainty = False

    def predict(
        self,
        image,
        return_raw: bool = False,
    ) -> Tuple[float, float, Optional[dict]]:
        """Run inference with all smart features.

        Returns:
            Tuple of (steering, throttle, metadata).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        preprocessed = self.preprocessor(image)

        # Get prediction
        with torch.no_grad():
            if self._model_has_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
                mean_pred, aleatoric, epistemic = self.model.predict_with_uncertainty(
                    preprocessed, n_samples=5
                )
                output = mean_pred.clamp(-1.0, 1.0).cpu().numpy().flatten()
                uncertainty = float(epistemic.mean().item())
            else:
                output = self.model(preprocessed)
                output = output.clamp(-1.0, 1.0).cpu().numpy().flatten()
                uncertainty = None

        raw_steering = float(output[0])
        raw_throttle = float(output[1])

        if return_raw:
            return raw_steering, raw_throttle, {"uncertainty": uncertainty}

        # Apply temporal smoothing
        if self.smoother:
            steering, throttle = self.smoother.smooth(raw_steering, raw_throttle)
        else:
            steering, throttle = raw_steering, raw_throttle

        # Apply adaptive throttle
        if self.throttle_controller:
            throttle = self.throttle_controller.compute_throttle(
                throttle, steering, uncertainty
            )

        # Check safety
        metadata = {"uncertainty": uncertainty, "safe": True, "reason": "OK"}
        if self.safety_monitor:
            is_safe, reason = self.safety_monitor.check(steering, uncertainty)
            metadata["safe"] = is_safe
            metadata["reason"] = reason

            if not is_safe:
                logger.warning(f"Safety warning: {reason}")
                # Reduce speed when unsafe
                throttle *= 0.5

        return steering, throttle, metadata

    def reset(self) -> None:
        """Reset all stateful components."""
        if self.smoother:
            self.smoother.reset()
        if self.safety_monitor:
            self.safety_monitor.reset()


class AutopilotController:
    """Main controller for autonomous driving with smart features."""

    def __init__(
        self,
        config: Optional[Config] = None,
        use_tensorrt: bool = True,
        use_smart_features: bool = True,
    ) -> None:
        self.config = config or get_default_config()
        self.engine = InferenceEngine(
            config=self.config,
            use_tensorrt=use_tensorrt,
            use_smoothing=use_smart_features,
            use_adaptive_throttle=use_smart_features,
            use_safety_monitor=use_smart_features,
        )

        self.car = None
        self.camera = None
        self._running = False

        # Performance tracking
        self._fps_history: Deque[float] = deque(maxlen=30)

    def setup(self, model_path: Optional[Path] = None) -> None:
        """Initialize hardware and load model."""

        self.engine.load_model(model_path)

        from jetracer.nvidia_racecar import NvidiaRacecar
        from jetcam.csi_camera import CSICamera

        self.car = NvidiaRacecar()
        self.car.throttle_gain = self.config.car.throttle_gain
        self.car.steering_offset = self.config.car.steering_offset

        self.camera = CSICamera(
            width=self.config.camera.width,
            height=self.config.camera.height,
        )

        logger.info("Autopilot controller initialized with smart features")

    def run(self) -> None:
        """Run the main control loop."""

        if self.car is None or self.camera is None:
            raise RuntimeError("Call setup() before run()")

        self._running = True
        self.engine.reset()
        logger.info("Starting autopilot control loop")

        try:
            while self._running:
                start_time = time.time()

                frame = self.camera.read()
                steering, throttle, metadata = self.engine.predict(frame)

                # Emergency stop if safety check fails repeatedly
                if not metadata.get("safe", True):
                    throttle = min(throttle, 0.3)

                self.car.steering = steering
                self.car.throttle = throttle

                # Track FPS
                elapsed = time.time() - start_time
                fps = 1.0 / (elapsed + 1e-6)
                self._fps_history.append(fps)

                if self.config.show_logs:
                    avg_fps = np.mean(list(self._fps_history))
                    uncertainty_str = (
                        f", unc={metadata['uncertainty']:.3f}"
                        if metadata.get('uncertainty') is not None
                        else ""
                    )
                    print(
                        f"\rFPS: {avg_fps:.0f}, "
                        f"Steer: {steering:+.3f}, "
                        f"Throttle: {throttle:+.3f}"
                        f"{uncertainty_str}, "
                        f"Status: {metadata.get('reason', 'OK')[:20]}",
                        end="",
                    )

        except KeyboardInterrupt:
            logger.info("Stopping autopilot")

        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the car safely."""
        self._running = False

        if self.car is not None:
            # Gradual stop
            for _ in range(5):
                self.car.throttle *= 0.5
                time.sleep(0.05)
            self.car.throttle = 0.0
            self.car.steering = 0.0
            logger.info("Car stopped safely")
