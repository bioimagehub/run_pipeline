"""
Train centroid U-Net with PyTorch Lightning.

Reads paired input/target images discovered via glob patterns and trains a
U-Net on them.  Input and target files are matched by the wildcard portion of
their filenames using ``rp.get_grouped_files_to_process``.

Input/output folder structure
------------------------------
<input-search-pattern>   e.g. training_set/input/*.tif
<target-search-pattern>  e.g. training_set/target-distance/*.tif
    model_training/                         (--output-folder, default: next to input)
        checkpoints/
            epoch001-val_loss0.0012.ckpt
            last.ckpt
        centroid_unet/                      <- TensorBoard logs

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

# Import utilities from the companion file in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from deep_learning_00_utils import (  # noqa: E402
    PairedPathDataset,
    UNet,
    get_train_transform,
)

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class CentroidUNetModule(pl.LightningModule):
    """PyTorch Lightning wrapper for the centroid U-Net."""

    def __init__(self, lr: float = 1e-3, max_epochs: int = 200) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels=1, out_channels=1)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        img, tgt = batch
        pred = self(img)
        loss = self.loss_fn(pred, tgt)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, self.max_epochs // 2),
            gamma=0.5,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a centroid U-Net on paired input/target images. "
            "Input and target files are matched by the wildcard portion of "
            "their filenames (via rp.get_grouped_files_to_process)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Train on Gaussian-blob targets (defaults)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_03_train.py'
  - --input-search-pattern: '%YAML%/training_set/input/*.tif'
  - --target-search-pattern: '%YAML%/training_set/target/*.tif'

- name: Train on distance-transform targets
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_03_train.py'
  - --input-search-pattern: '%YAML%/training_set/input/*.tif'
  - --target-search-pattern: '%YAML%/training_set/target-distance/*.tif'
  - --output-folder: '%YAML%/model_training_distance'
  - --max-epochs: 200
  - --batch-size: 16
  - --lr: 5e-4

- name: Pause to inspect TensorBoard logs
  type: pause
  message: 'Run: tensorboard --logdir <output-folder>/centroid_unet'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help=(
            "Glob pattern for input image files. Must contain '*' so files "
            "can be matched to targets by their wildcard basename "
            "(e.g. 'training_set/input/*.tif')."
        ),
    )
    parser.add_argument(
        "--target-search-pattern",
        required=True,
        help=(
            "Glob pattern for target image files. Must contain '*' so files "
            "can be matched to inputs by their wildcard basename "
            "(e.g. 'training_set/target-distance/*.tif')."
        ),
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help=(
            "Folder for checkpoints and TensorBoard logs. "
            "Defaults to a 'model_training' folder next to the input pattern directory."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data used for validation (default: 0.15).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--zero-keep-ratio",
        type=float,
        default=0.10,
        help=(
            "Fraction of non-zero targets to keep as empty (all-zero) pairs "
            "during training. 0.10 means keep ~10%% as many zero pairs as "
            "non-zero pairs. Set to 1.0 to disable filtering (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility and zero-pair downsampling (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------ #
    # Discover and pair input/target files                                 #
    # ------------------------------------------------------------------ #
    search_subfolders = "**" in args.input_search_pattern or "**" in args.target_search_pattern
    groups = rp.get_grouped_files_to_process(
        {"input": args.input_search_pattern, "target": args.target_search_pattern},
        search_subfolders=search_subfolders,
    )

    pairs: list[tuple[str, str]] = []
    skipped_basenames: list[str] = []
    for basename, files in groups.items():
        if "input" in files and "target" in files:
            pairs.append((files["input"], files["target"]))
        else:
            skipped_basenames.append(basename)

    if skipped_basenames:
        logger.warning(
            "%d basename(s) skipped (missing input or target): %s",
            len(skipped_basenames),
            skipped_basenames,
        )
    if not pairs:
        raise SystemExit(
            "No matched input/target pairs found.\n"
            f"  input pattern : {args.input_search_pattern}\n"
            f"  target pattern: {args.target_search_pattern}"
        )

    logger.info("Matched %d input/target pair(s)", len(pairs))
    for inp, tgt in pairs:
        logger.debug("  %s  <->  %s", inp, tgt)

    # ------------------------------------------------------------------ #
    # Filter empty (all-zero target) pairs                                #
    # ------------------------------------------------------------------ #
    if args.zero_keep_ratio < 1.0:
        nonzero_pairs: list[tuple[str, str]] = []
        zero_pairs: list[tuple[str, str]] = []
        for inp, tgt in pairs:
            if tifffile.imread(tgt).max() > 0:
                nonzero_pairs.append((inp, tgt))
            else:
                zero_pairs.append((inp, tgt))

        n_keep = max(1, int(round(len(nonzero_pairs) * args.zero_keep_ratio)))
        rng = np.random.default_rng(args.seed)
        kept_indices = rng.choice(
            len(zero_pairs), size=min(n_keep, len(zero_pairs)), replace=False
        )
        kept_zero_pairs = [zero_pairs[i] for i in sorted(kept_indices)]

        logger.info(
            "Zero-filtering: %d non-zero pairs, %d zero pairs total, "
            "keeping %d zero pairs (--zero-keep-ratio=%.2f)",
            len(nonzero_pairs), len(zero_pairs), len(kept_zero_pairs), args.zero_keep_ratio,
        )
        pairs = nonzero_pairs + kept_zero_pairs
        pairs.sort(key=lambda x: x[0])
    else:
        logger.info("Zero-filtering disabled (--zero-keep-ratio=1.0)")

    if not pairs:
        raise SystemExit("No pairs remain after zero-filtering. Lower --zero-keep-ratio or check targets.")

    # ------------------------------------------------------------------ #
    # Resolve output folder                                                #
    # ------------------------------------------------------------------ #
    if args.output_folder is not None:
        output_folder = Path(args.output_folder).resolve()
    else:
        # Default: sibling of the input pattern's directory
        input_dir = Path(args.input_search_pattern.split("*")[0]).resolve()
        if not input_dir.is_dir():
            input_dir = input_dir.parent
        output_folder = input_dir.parent / "model_training"

    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Output folder: %s", output_folder)

    # ------------------------------------------------------------------ #
    # Seed everything                                                      #
    # ------------------------------------------------------------------ #
    pl.seed_everything(args.seed, workers=True)

    # ------------------------------------------------------------------ #
    # Datasets & data loaders                                              #
    # ------------------------------------------------------------------ #
    full_ds = PairedPathDataset(pairs, transform=None)
    n_total = len(full_ds)

    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    logger.info("Dataset: %d total, %d train, %d val", n_total, n_train, n_val)

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Apply augmentation to training subset only
    train_ds.dataset = PairedPathDataset(pairs, transform=get_train_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = CentroidUNetModule(lr=args.lr, max_epochs=args.max_epochs)

    # ------------------------------------------------------------------ #
    # Callbacks & logger                                                   #
    # ------------------------------------------------------------------ #
    tb_logger = TensorBoardLogger(
        save_dir=str(output_folder),
        name="centroid_unet",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_folder / "checkpoints"),
        filename="epoch{epoch:03d}-val_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ------------------------------------------------------------------ #
    # Trainer                                                              #
    # ------------------------------------------------------------------ #
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=1,
        deterministic=True,
    )

    tb_log_dir = output_folder / "centroid_unet"
    logger.info("")
    logger.info("=" * 60)
    logger.info("TensorBoard: open a separate terminal and run:")
    logger.info("  tensorboard --logdir \"%s\"", tb_log_dir)
    logger.info("Then open: http://localhost:6006")
    logger.info("=" * 60)
    logger.info("")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("Best checkpoint : %s", checkpoint_cb.best_model_path)
    logger.info("TensorBoard logs: %s", tb_log_dir)
    logger.info("  -> tensorboard --logdir \"%s\"", tb_log_dir)


if __name__ == "__main__":
    main()
