"""Image preprocessing utilities for autopilot inference."""

from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    """Crop the center square from a rectangular image.

    Args:
        frame: Input image as numpy array (H, W, C).

    Returns:
        Square cropped image.
    """
    height, width = frame.shape[:2]

    if width > height:
        square_size = height
        offset = (width - square_size) // 2
        return frame[:, offset:offset + square_size]
    else:
        square_size = width
        offset = (height - square_size) // 2
        return frame[offset:offset + square_size, :]


class ImagePreprocessor:
    """Efficient image preprocessing for inference.

    Caches normalization tensors on the target device for speed.
    """

    def __init__(self, device: torch.device, frame_size: int = 224) -> None:
        self.device = device
        self.frame_size = frame_size

        self.mean = IMAGENET_MEAN.to(device)[:, None, None]
        self.std = IMAGENET_STD.to(device)[:, None, None]

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a BGR image for model inference.

        Args:
            image: BGR image from camera (H, W, 3).

        Returns:
            Preprocessed tensor ready for model (1, 3, H, W).
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = center_crop_square(image)
        image = cv2.resize(image, (self.frame_size, self.frame_size))

        pil_image = Image.fromarray(image)
        tensor = TF.to_tensor(pil_image).to(self.device)

        tensor = (tensor - self.mean) / self.std

        return tensor.unsqueeze(0)

    def denormalize(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a normalized tensor back to a PIL image.

        Useful for visualization during testing.
        """
        tensor = tensor.cpu().clone()

        mean = IMAGENET_MEAN[:, None, None]
        std = IMAGENET_STD[:, None, None]
        tensor = tensor * std + mean

        tensor = tensor.clamp(0, 1)

        return TF.to_pil_image(tensor)
