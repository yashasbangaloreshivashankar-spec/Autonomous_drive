"""Command-line interface for the Jetson Autopilot system."""

import argparse
import logging
from pathlib import Path
import sys

from .config import Config, get_default_config
from .model import AutopilotModel
from .dataset import create_data_loaders
from .trainer import Trainer
from .inference import AutopilotController


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def train(args: argparse.Namespace) -> None:
    """Train the autopilot model."""

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.config:
        config = Config.from_json(Path(args.config))
    else:
        config = get_default_config()

    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.lr:
        config.training.initial_lr = args.lr
    if args.model_name:
        config.model_name = args.model_name
    if args.models_dir:
        config.models_dir = Path(args.models_dir)

    training_dir = Path(args.training_dir) if args.training_dir else None
    validation_dir = Path(args.validation_dir) if args.validation_dir else None
    testing_dir = Path(args.testing_dir) if args.testing_dir else None

    if training_dir is None or validation_dir is None:
        logger.error("Training and validation directories are required")
        sys.exit(1)

    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        training_dir=training_dir,
        validation_dir=validation_dir,
        testing_dir=testing_dir,
    )

    model = AutopilotModel(config=config.model, pretrained=True)
    trainer = Trainer(model=model, config=config)

    history = trainer.train(train_loader, val_loader)

    logger.info(f"Training complete. Final losses - "
                f"train: {history['training_loss'][-1]:.6f}, "
                f"val: {history['validation_loss'][-1]:.6f}")

    if test_loader is not None:
        logger.info("Running test evaluation...")
        avg_loss, results = trainer.test(test_loader, config.model_path)

        for result in results:
            status = "PASS" if result["passed"] else "FAIL"
            logger.info(
                f"[{status}] {result['name']}: "
                f"expected={result['expected']}, "
                f"predicted={[f'{v:.3f}' for v in result['predicted']]}, "
                f"loss={result['loss']:.4f}"
            )


def run(args: argparse.Namespace) -> None:
    """Run the autopilot on the car."""

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.config:
        config = Config.from_json(Path(args.config))
    else:
        config = get_default_config()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = config.model_path

    config.show_logs = args.show_fps

    if args.throttle_gain:
        config.car.throttle_gain = args.throttle_gain
    if args.steering_offset:
        config.car.steering_offset = args.steering_offset

    controller = AutopilotController(
        config=config,
        use_tensorrt=not args.no_tensorrt,
    )

    controller.setup(model_path)
    controller.run()


def test(args: argparse.Namespace) -> None:
    """Test the model on a test dataset."""

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.config:
        config = Config.from_json(Path(args.config))
    else:
        config = get_default_config()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = config.model_path

    testing_dir = Path(args.testing_dir)

    _, _, test_loader = create_data_loaders(
        config=config,
        testing_dir=testing_dir,
    )

    model = AutopilotModel(config=config.model, pretrained=False)
    trainer = Trainer(model=model, config=config)

    avg_loss, results = trainer.test(test_loader, model_path)

    passed = sum(1 for r in results if r["passed"])
    print(f"\nResults: {passed}/{len(results)} passed, Average Loss: {avg_loss:.4f}\n")

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['name']}: "
            f"loss={result['loss']:.4f}, "
            f"expected={result['expected']}, "
            f"predicted={[f'{v:.3f}' for v in result['predicted']]}"
        )


def init_config(args: argparse.Namespace) -> None:
    """Generate a default configuration file."""

    config = get_default_config()
    output_path = Path(args.output)
    config.to_json(output_path)
    print(f"Configuration saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jetson Autopilot - Self-driving toy car system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the autopilot model")
    train_parser.add_argument("--training-dir", "-t", required=True, help="Training dataset directory")
    train_parser.add_argument("--validation-dir", "-v", required=True, help="Validation dataset directory")
    train_parser.add_argument("--testing-dir", help="Testing dataset directory (optional)")
    train_parser.add_argument("--config", "-c", help="Path to config JSON file")
    train_parser.add_argument("--models-dir", help="Directory to save models")
    train_parser.add_argument("--model-name", help="Name for the model")
    train_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    train_parser.add_argument("--epochs", "-e", type=int, help="Maximum epochs")
    train_parser.add_argument("--lr", type=float, help="Initial learning rate")
    train_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    train_parser.set_defaults(func=train)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the autopilot on the car")
    run_parser.add_argument("--model-path", "-m", help="Path to model checkpoint")
    run_parser.add_argument("--config", "-c", help="Path to config JSON file")
    run_parser.add_argument("--no-tensorrt", action="store_true", help="Disable TensorRT optimization")
    run_parser.add_argument("--show-fps", action="store_true", help="Show FPS and control values")
    run_parser.add_argument("--throttle-gain", type=float, help="Throttle gain multiplier")
    run_parser.add_argument("--steering-offset", type=float, help="Steering offset correction")
    run_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    run_parser.set_defaults(func=run)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the model on a dataset")
    test_parser.add_argument("--testing-dir", "-t", required=True, help="Testing dataset directory")
    test_parser.add_argument("--model-path", "-m", help="Path to model checkpoint")
    test_parser.add_argument("--config", "-c", help="Path to config JSON file")
    test_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    test_parser.set_defaults(func=test)

    # Init command
    init_parser = subparsers.add_parser("init", help="Generate default config file")
    init_parser.add_argument("--output", "-o", default="config.json", help="Output path")
    init_parser.set_defaults(func=init_config)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
