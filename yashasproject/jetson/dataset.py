"""Dataset handling for autopilot training with advanced augmentations."""

from pathlib import Path
from typing import Optional, Tuple, List, Callable
import csv
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageDraw

from .config import Config, AugmentationConfig, get_default_config
from .preprocessing import center_crop_square


class AdvancedAugmentations:
    """Advanced augmentation techniques for driving data."""

    @staticmethod
    def add_shadow(image: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Add random shadow to simulate varying lighting."""
        width, height = image.size

        # Random shadow polygon
        x1 = random.randint(0, width)
        x2 = random.randint(0, width)

        vertices = [
            (x1, 0),
            (x2, 0),
            (x2 + random.randint(-100, 100), height),
            (x1 + random.randint(-100, 100), height),
        ]

        # Create shadow mask
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        draw.polygon(vertices, fill=int(255 * (1 - intensity)))

        # Apply shadow
        result = image.copy()
        result.putalpha(mask)
        background = Image.new('RGB', (width, height), (0, 0, 0))
        background.paste(result, mask=mask)

        return Image.blend(image, background, intensity * 0.5)

    @staticmethod
    def add_perspective_transform(
        image: Image.Image,
        steering: float,
        intensity: float = 0.1,
    ) -> Tuple[Image.Image, float]:
        """Apply perspective transform to simulate different viewing angles."""
        width, height = image.size

        # Random perspective shift
        shift = random.uniform(-intensity, intensity) * width

        # Define source and destination points
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        dst_points = np.float32([
            [shift, 0],
            [width + shift, 0],
            [width - shift, height],
            [-shift, height]
        ])

        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transform
        img_array = np.array(image)
        transformed = cv2.warpPerspective(img_array, matrix, (width, height))

        # Adjust steering based on perspective shift
        steering_adjustment = shift / width * 0.5
        new_steering = np.clip(steering + steering_adjustment, -1, 1)

        return Image.fromarray(transformed), new_steering

    @staticmethod
    def cutout(image: Image.Image, n_holes: int = 1, size: int = 40) -> Image.Image:
        """Apply cutout augmentation (random erasing)."""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        for _ in range(n_holes):
            y = random.randint(0, height - size)
            x = random.randint(0, width - size)

            img_array[y:y+size, x:x+size] = 128  # Gray fill

        return Image.fromarray(img_array)

    @staticmethod
    def motion_blur(image: Image.Image, kernel_size: int = 15) -> Image.Image:
        """Apply motion blur to simulate fast movement."""
        img_array = np.array(image)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1
        kernel = kernel / kernel_size

        blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(blurred)

    @staticmethod
    def brightness_gradient(image: Image.Image) -> Image.Image:
        """Apply non-uniform brightness (simulates sun glare)."""
        width, height = image.size

        # Create gradient mask
        gradient = np.linspace(0.7, 1.3, width)
        gradient = np.tile(gradient, (height, 1))

        if random.random() > 0.5:
            gradient = gradient.T
            gradient = np.tile(gradient[:, 0:1], (1, width))

        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * gradient[:, :, np.newaxis]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)


class AutopilotDataset(Dataset):
    """PyTorch Dataset with advanced augmentations for autopilot training."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        directory: Path,
        frame_size: int = 224,
        augmentation: Optional[AugmentationConfig] = None,
        keep_in_memory: bool = False,
        use_advanced_augmentation: bool = True,
    ) -> None:
        super().__init__()

        self.directory = Path(directory)
        self.frame_size = frame_size
        self.augmentation = augmentation
        self.keep_in_memory = keep_in_memory
        self.use_advanced_augmentation = use_advanced_augmentation and augmentation is not None

        self.samples: List[Tuple[str, any, float, float]] = []
        self._load_annotations()

        self._base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
        ])

        self.advanced_aug = AdvancedAugmentations()

    def _load_annotations(self) -> None:
        annotations_path = self.directory / "annotations.csv"

        with open(annotations_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue

                name = row[0].strip()
                steering = float(row[1].strip())
                throttle = float(row[2].strip())

                image_path = self.directory / f"{name}.jpg"
                if not image_path.exists() or image_path.stat().st_size == 0:
                    continue

                if self.keep_in_memory:
                    image = self._load_image(image_path)
                    self.samples.append((name, image, steering, throttle))
                else:
                    self.samples.append((name, image_path, steering, throttle))

        print(f"Loaded dataset with {len(self.samples)} samples from {self.directory}")

    def _load_image(self, path: Path) -> Image.Image:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = center_crop_square(image)
        image = cv2.resize(image, (self.frame_size, self.frame_size))
        return Image.fromarray(image)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        name, image_data, steering, throttle = self.samples[idx]

        if not self.keep_in_memory:
            image = self._load_image(image_data)
        else:
            image = image_data.copy()

        image, steering = self._apply_augmentation(image, steering)
        image = self._base_transform(image)

        return name, image, torch.tensor([steering, throttle], dtype=torch.float32)

    def _apply_augmentation(
        self,
        image: Image.Image,
        steering: float,
    ) -> Tuple[Image.Image, float]:
        if self.augmentation is None:
            return image, steering

        aug = self.augmentation

        # Basic augmentations
        if aug.random_blur and random.random() > 0.5:
            image = image.filter(ImageFilter.BLUR)

        if aug.random_noise and random.random() > 0.5:
            image = self._add_salt_pepper_noise(image, aug.noise_amount)

        if aug.random_horizontal_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            steering = -steering

        if aug.random_color_jitter:
            jitter = T.ColorJitter(
                brightness=aug.color_jitter_brightness,
                contrast=aug.color_jitter_contrast,
                hue=aug.color_jitter_hue,
                saturation=aug.color_jitter_saturation,
            )
            image = jitter(image)

        # Advanced augmentations
        if self.use_advanced_augmentation:
            if random.random() > 0.7:
                image = self.advanced_aug.add_shadow(image, intensity=random.uniform(0.2, 0.5))

            if random.random() > 0.8:
                image, steering = self.advanced_aug.add_perspective_transform(
                    image, steering, intensity=0.08
                )

            if random.random() > 0.85:
                image = self.advanced_aug.cutout(image, n_holes=random.randint(1, 3), size=30)

            if random.random() > 0.9:
                image = self.advanced_aug.motion_blur(image, kernel_size=random.choice([5, 7, 9]))

            if random.random() > 0.8:
                image = self.advanced_aug.brightness_gradient(image)

        return image, steering

    @staticmethod
    def _add_salt_pepper_noise(image: Image.Image, amount: float = 0.1) -> Image.Image:
        arr = np.array(image, dtype=np.float32)

        num_salt = int(np.ceil(amount * arr.size * 0.5))
        salt_coords = tuple(np.random.randint(0, dim, num_salt) for dim in arr.shape)
        arr[salt_coords] = 255.0

        num_pepper = int(np.ceil(amount * arr.size * 0.5))
        pepper_coords = tuple(np.random.randint(0, dim, num_pepper) for dim in arr.shape)
        arr[pepper_coords] = 0.0

        return Image.fromarray(arr.astype(np.uint8))


class MixupDataset(Dataset):
    """Wrapper dataset that applies Mixup augmentation."""

    def __init__(self, dataset: AutopilotDataset, alpha: float = 0.2) -> None:
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        name1, img1, target1 = self.dataset[idx]

        if random.random() > 0.5:
            # Apply mixup
            idx2 = random.randint(0, len(self.dataset) - 1)
            name2, img2, target2 = self.dataset[idx2]

            lam = np.random.beta(self.alpha, self.alpha)
            img = lam * img1 + (1 - lam) * img2
            target = lam * target1 + (1 - lam) * target2

            return name1, img, target

        return name1, img1, target1


def create_data_loaders(
    config: Optional[Config] = None,
    training_dir: Optional[Path] = None,
    validation_dir: Optional[Path] = None,
    testing_dir: Optional[Path] = None,
    use_mixup: bool = True,
    num_workers: int = 4,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create data loaders for training, validation, and testing."""

    if config is None:
        config = get_default_config()

    loaders = []

    for directory, use_augmentation, batch_size, is_training in [
        (training_dir, True, config.training.batch_size, True),
        (validation_dir, False, config.training.batch_size, False),
        (testing_dir, False, 1, False),
    ]:
        if directory is None:
            loaders.append(None)
            continue

        augmentation = config.augmentation if use_augmentation else None

        dataset = AutopilotDataset(
            directory=directory,
            frame_size=config.model.frame_size,
            augmentation=augmentation,
            keep_in_memory=True,
            use_advanced_augmentation=use_augmentation,
        )

        # Apply mixup for training
        if is_training and use_mixup:
            dataset = MixupDataset(dataset, alpha=0.2)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers if not is_training else 0,  # Avoid issues with in-memory data
            pin_memory=True,
            drop_last=is_training and len(dataset) > batch_size,
        )
        loaders.append(loader)

    return tuple(loaders)
