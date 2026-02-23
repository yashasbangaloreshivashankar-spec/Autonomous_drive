"""Data collection utilities for recording training data."""

import csv
import os
import shutil
import time
from pathlib import Path
from typing import Optional
import logging

import cv2

from .config import Config, get_default_config

logger = logging.getLogger(__name__)


class DataCollector:
    """Handles recording and saving training data from the car."""

    def __init__(
        self,
        config: Optional[Config] = None,
        dataset_name: str = "dataset",
        dataset_mode: str = "training",
    ) -> None:
        self.config = config or get_default_config()

        self.datasets_dir = self.config.datasets_dir
        self.tmp_dir = self.datasets_dir / "tmp"
        self.output_dir = self.datasets_dir / f"{dataset_name}_{dataset_mode}"

        self.tmp_annotations = self.tmp_dir / "annotations.csv"
        self.main_annotations = self.output_dir / "annotations.csv"

        self.car = None
        self.camera = None
        self.controller = None

        self._recording = False
        self._frame_count = 0

    def setup(self) -> None:
        """Initialize hardware for data collection."""

        from jetracer.nvidia_racecar import NvidiaRacecar
        from jetcam.csi_camera import CSICamera

        self.car = NvidiaRacecar()
        self.car.throttle_gain = self.config.car.throttle_gain
        self.car.steering_offset = self.config.car.steering_offset

        self.camera = CSICamera(
            width=self.config.camera.width,
            height=self.config.camera.height,
            capture_fps=self.config.camera.fps,
        )
        self.camera.running = True

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reset_tmp_dir()

        logger.info("Data collector initialized")

    def _reset_tmp_dir(self) -> None:
        """Clear and recreate the temporary directory."""
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)

    def start_recording(self) -> None:
        """Start a new recording session."""
        self._reset_tmp_dir()
        self._recording = True
        self._frame_count = 0
        logger.info("Recording started")

    def stop_recording(self) -> None:
        """Stop the current recording session."""
        self._recording = False
        logger.info(f"Recording stopped. Frames captured: {self._frame_count}")

    def save_recording(self) -> None:
        """Save recorded data to the main dataset directory."""

        if not self.tmp_dir.exists():
            logger.warning("No temporary data to save")
            return

        for file in self.tmp_dir.iterdir():
            if file.suffix == ".csv":
                if self.main_annotations.exists() and self.main_annotations.stat().st_size > 0:
                    with open(self.main_annotations, "a") as main_f:
                        with open(file) as tmp_f:
                            main_f.write(tmp_f.read())
                else:
                    shutil.copy(file, self.main_annotations)
            else:
                shutil.move(str(file), str(self.output_dir / file.name))

        self._reset_tmp_dir()
        logger.info(f"Recording saved to {self.output_dir}")

    def discard_recording(self) -> None:
        """Discard the current recording without saving."""
        self._reset_tmp_dir()
        logger.info("Recording discarded")

    def capture_frame(self) -> None:
        """Capture and save current frame with annotations."""

        if not self._recording or self.camera is None or self.car is None:
            return

        frame = self.camera.read()
        if frame is None:
            return

        timestamp = str(int(time.time() * 1000))
        image_path = self.tmp_dir / f"{timestamp}.jpg"

        cv2.imwrite(str(image_path), frame)

        with open(self.tmp_annotations, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                f"{self.car.steering:.3f}",
                f"{self.car.throttle:.3f}",
            ])

        self._frame_count += 1


def setup_gamepad_controls(collector: DataCollector) -> None:
    """Set up gamepad controls for data collection (for Jupyter notebook use).

    Button mappings (Gamepad Mode 2):
    - Left stick: Throttle
    - Right stick: Steering
    - Button X (2): Start recording
    - Button B (1): Save recording
    - Button RB (5): Half throttle
    - Button RT (7): Brake
    """

    import ipywidgets
    import traitlets

    controller = ipywidgets.widgets.Controller(index=0)

    def clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        return min(max_val, max(min_val, value))

    def is_pressed(change) -> bool:
        return change["name"] == "pressed" and change["new"]

    traitlets.dlink(
        (controller.axes[2], "value"),
        (collector.car, "steering"),
        transform=lambda x: clamp(-x),
    )

    traitlets.dlink(
        (controller.axes[1], "value"),
        (collector.car, "throttle"),
        transform=lambda x: clamp(x),
    )

    traitlets.dlink(
        (controller.buttons[7], "value"),
        (collector.car, "throttle"),
        transform=lambda x: 0.0,
    )

    traitlets.dlink(
        (controller.buttons[5], "value"),
        (collector.car, "throttle"),
        transform=lambda x: -0.5 if x > 0.5 else 0,
    )

    controller.buttons[2].observe(
        lambda x: collector.start_recording() if is_pressed(x) else None
    )
    controller.buttons[1].observe(
        lambda x: collector.save_recording() if is_pressed(x) else None
    )

    return controller
