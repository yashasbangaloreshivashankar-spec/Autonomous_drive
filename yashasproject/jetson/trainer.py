"""Advanced training logic with modern techniques."""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm

from .config import Config, get_default_config
from .model import AutopilotModel, GaussianNLLLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping with patience and minimum delta."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = 'min',
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class GradientClipping:
    """Gradient clipping utility."""

    def __init__(self, max_norm: float = 1.0) -> None:
        self.max_norm = max_norm

    def __call__(self, model: nn.Module) -> float:
        return nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


class Trainer:
    """Advanced trainer with modern deep learning techniques.

    Features:
    - AdamW optimizer with weight decay
    - Cosine annealing with warm restarts OR OneCycle learning rate
    - Gradient clipping
    - Mixed precision training (if available)
    - Uncertainty-aware loss
    - Exponential moving average of weights
    """

    def __init__(
        self,
        model: AutopilotModel,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        use_ema: bool = True,
    ) -> None:
        self.config = config or get_default_config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.use_ema = use_ema

        self.model = model.to(self.device)

        # Uncertainty-aware loss if model supports it
        if model.use_uncertainty:
            self.loss_fn = GaussianNLLLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # AdamW with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.initial_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        self.scheduler = None  # Will be set in train()
        self.gradient_clipper = GradientClipping(max_norm=1.0)

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Exponential moving average
        if self.use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = 0.999
        else:
            self.ema_model = None

        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.learning_rates: List[float] = []

    def _create_ema_model(self) -> AutopilotModel:
        """Create EMA copy of model."""
        ema = AutopilotModel(
            config=self.model.config,
            pretrained=False,
            use_attention=self.model.use_attention,
            use_uncertainty=self.model.use_uncertainty,
            use_multi_scale=self.model.use_multi_scale,
        )
        ema.load_state_dict(self.model.state_dict())
        ema.to(self.device)
        ema.eval()
        for param in ema.parameters():
            param.requires_grad = False
        return ema

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[Path] = None,
        use_one_cycle: bool = True,
    ) -> Dict[str, List[float]]:
        """Run the full training loop with advanced techniques."""

        if save_path is None:
            save_path = self.config.model_path

        # Setup learning rate scheduler
        total_steps = len(train_loader) * self.config.training.max_epochs

        if use_one_cycle:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.initial_lr * 10,
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=100,
            )
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=len(train_loader) * 5,  # Restart every 5 epochs
                T_mult=2,
                eta_min=1e-7,
            )

        early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=1e-6,
        )

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Using AMP: {self.use_amp}, EMA: {self.use_ema}")

        best_val_loss = float("inf")

        for epoch in range(self.config.training.max_epochs):
            train_loss = self._run_epoch(train_loader, training=True, epoch=epoch)
            self.training_losses.append(train_loss)

            # Use EMA model for validation if available
            val_model = self.ema_model if self.ema_model else self.model
            val_loss = self._run_epoch(val_loader, training=False, epoch=epoch, model=val_model)
            self.validation_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"Validation loss improved to {val_loss:.6f}, saving model")

                # Save EMA model if available, otherwise regular model
                save_model = self.ema_model if self.ema_model else self.model
                save_model.save(save_path)

            logger.info(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, "
                f"best={best_val_loss:.6f}, "
                f"lr={current_lr:.2e}"
            )

            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered. Best loss: {best_val_loss:.6f}")
                break

        return {
            "training_loss": self.training_losses,
            "validation_loss": self.validation_losses,
            "learning_rate": self.learning_rates,
        }

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int,
        model: Optional[AutopilotModel] = None,
    ) -> float:
        """Run a single training or validation epoch."""

        if model is None:
            model = self.model

        if training:
            model.train()
            desc = f"Training Epoch {epoch + 1}"
        else:
            model.eval()
            desc = f"Validation Epoch {epoch + 1}"

        total_loss = 0.0
        num_batches = 0

        with tqdm(loader, desc=desc, unit="batch") as pbar:
            for _, images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                if training:
                    loss = self._training_step(images, targets)
                else:
                    loss = self._validation_step(images, targets, model)

                total_loss += loss
                num_batches += 1

                pbar.set_postfix(loss=f"{loss:.4f}")

        return total_loss / num_batches

    def _training_step(self, images: torch.Tensor, targets: torch.Tensor) -> float:
        """Single training step with mixed precision."""

        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(images, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.gradient_clipper(self.model)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(images, targets)
            loss.backward()
            self.gradient_clipper(self.model)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self._update_ema()

        return loss.item()

    def _validation_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        model: AutopilotModel,
    ) -> float:
        """Single validation step."""

        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(images, targets, model)
            else:
                loss = self._compute_loss(images, targets, model)

        return loss.item()

    def _compute_loss(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[AutopilotModel] = None,
    ) -> torch.Tensor:
        """Compute loss with uncertainty if available."""

        if model is None:
            model = self.model

        if model.use_uncertainty:
            mean, log_var = model.output(model.head(model._forward_backbone(images)))
            return self.loss_fn(mean, log_var, targets)
        else:
            outputs = model(images)
            return self.loss_fn(outputs, targets)

    def test(
        self,
        test_loader: DataLoader,
        model_path: Optional[Path] = None,
        use_uncertainty: bool = True,
    ) -> Tuple[float, List[Dict]]:
        """Evaluate model on test set with uncertainty estimation."""

        if model_path is not None:
            self.model = AutopilotModel.from_checkpoint(model_path, self.device)
            self.model.to(self.device)

        self.model.eval()

        results = []
        total_loss = 0.0
        threshold = self.config.training.acceptable_testing_loss
        mse_loss = nn.MSELoss()

        for name, image, target in test_loader:
            image = image.to(self.device)
            target = target.to(self.device)

            if use_uncertainty and self.model.use_uncertainty:
                # Get uncertainty estimates
                mean_pred, aleatoric, epistemic = self.model.predict_with_uncertainty(
                    image, n_samples=10
                )
                prediction = mean_pred.clamp(-1, 1)
                uncertainty = (aleatoric.mean().item() if aleatoric is not None else 0,
                               epistemic.mean().item())
            else:
                with torch.no_grad():
                    prediction = self.model(image).clamp(-1, 1)
                uncertainty = None

            loss = mse_loss(prediction, target).item()

            results.append({
                "name": name[0],
                "expected": target.cpu().numpy().flatten().tolist(),
                "predicted": prediction.cpu().numpy().flatten().tolist(),
                "loss": loss,
                "passed": loss < threshold,
                "uncertainty": uncertainty,
            })

            total_loss += loss

        avg_loss = total_loss / len(results) if results else 0.0
        passed = sum(1 for r in results if r["passed"])

        logger.info(f"Test Results: {passed}/{len(results)} passed, avg_loss={avg_loss:.4f}")

        return avg_loss, results
