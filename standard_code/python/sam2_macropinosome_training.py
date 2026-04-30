"""
sam2_macropinosome_training.py
------------------------------
Fine-tune SAM2 for macropinosome segmentation using parameter-efficient adapters.

Strategy (from BIPHUB discussion):
  1. Shared components (image encoder, prompt encoder, mask decoder) are the
     primary adaptation targets — updates propagate into both image and video pipelines.
  2. LoRA (Low-Rank Adaptation) is injected into the image encoder's attention
     linear layers so box-based signals update visual representations.
  3. The small mask decoder and prompt encoder are fully fine-tuned (they are
     already compact: ~3–6 M params combined).
  4. A lightweight per-frame refinement CNN head consumes
     [image, coarse_sam2_mask, box_mask] and outputs a refined mask logit.
     Because it mirrors the per-frame stage of the SAM2 video pipeline, gains
     transfer immediately into video tracking/fusion.

Training data comes from deep_learning_02_make_training_set_sam2.py:
  <training-folder>/
      training_set/
          input/          float32 image patches (.tif)
          target_sam2/    float32 binary masks (.tif, values 0.0 or 1.0)

Boxes are derived automatically from each target mask (bounding box of the
non-zero region plus a small pixel margin).

Saved artifacts:
  <output-folder>/
      checkpoints/            PyTorch Lightning model checkpoints
      sam2_adapter_logs/      TensorBoard event files

Loss functions used:
  - BCE + Dice loss (primary)
  - Boundary/edge loss (optional, --boundary-loss)
  - Auxiliary BCE + Dice on coarse SAM2 mask (0.3 weight, always)

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MacropinosomeDataset(torch.utils.data.Dataset):
    """Load paired image patches and SAM2 binary mask patches.

    Files are matched by filename stem (e.g. ``0.tif`` ↔ ``0.tif``).
    Each item returns:

    - ``image_rgb``  : (3, H, W) float32 tensor, normalised to [0, 1]
    - ``box_xyxy``   : (4,) float32 tensor, absolute pixel XYXY coords
    - ``gt_mask``    : (1, H, W) float32 binary mask tensor

    Args:
        input_dir:  Directory with float32 image patch .tif files.
        mask_dir:   Directory with float32 binary mask .tif files.
        patch_size: If > 0, resize every patch (and box) to this square size.
        augment:    Apply random horizontal/vertical flips.
        box_margin: Extra pixels added around the bounding box on each side.
    """

    def __init__(
        self,
        input_dir: str | Path,
        mask_dir: str | Path,
        patch_size: int = 0,
        augment: bool = False,
        box_margin: int = 4,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.box_margin = box_margin

        input_stems = {Path(f).stem for f in self.input_dir.glob("*.tif")}
        mask_stems = {Path(f).stem for f in self.mask_dir.glob("*.tif")}
        common = sorted(input_stems & mask_stems)

        if not common:
            raise FileNotFoundError(
                f"No matching .tif pairs found in:\n  {input_dir}\n  {mask_dir}"
            )

        self.stems = common
        logger.info("Dataset: %d image-mask pairs loaded.", len(self.stems))

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img = tifffile.imread(self.input_dir / f"{stem}.tif").astype(np.float32)
        mask = tifffile.imread(self.mask_dir / f"{stem}.tif").astype(np.float32)

        # Collapse extra dimensions to 2-D (HxW)
        while img.ndim > 2:
            img = img[0]
        while mask.ndim > 2:
            mask = mask[0]

        # Normalise image to [0, 1]
        lo = float(np.percentile(img, 0.1))
        hi = float(np.percentile(img, 99.9))
        img = np.clip((img - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

        # Binarise mask
        mask = (mask > 0.5).astype(np.float32)

        H, W = img.shape

        # Optional resize
        if self.patch_size > 0 and (H != self.patch_size or W != self.patch_size):
            img = _resize_array(img, self.patch_size)
            mask = _resize_array(mask, self.patch_size, is_mask=True)
            H = W = self.patch_size

        # Derive bounding box from mask
        nz_rows, nz_cols = np.where(mask > 0.5)
        m = self.box_margin
        if len(nz_rows) > 0:
            y1 = max(0, int(nz_rows.min()) - m)
            y2 = min(H, int(nz_rows.max()) + m + 1)
            x1 = max(0, int(nz_cols.min()) - m)
            x2 = min(W, int(nz_cols.max()) + m + 1)
        else:
            y1, x1, y2, x2 = 0, 0, H, W

        # Greyscale → replicated RGB: (3, H, W)
        img_rgb = np.stack([img, img, img], axis=0)

        image_t = torch.from_numpy(img_rgb)
        box_t = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        if self.augment:
            image_t, mask_t, box_t = _random_flip(image_t, mask_t, box_t, H, W)

        return image_t, box_t, mask_t


def _resize_array(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    """Resize a (H, W) numpy array to (size, size) using PIL."""
    from PIL import Image as PILImage

    mode = "L"
    pil = PILImage.fromarray((arr * 255).astype(np.uint8), mode=mode)
    interp = PILImage.NEAREST if is_mask else PILImage.BILINEAR
    pil = pil.resize((size, size), interp)
    result = np.array(pil, dtype=np.float32) / 255.0
    if is_mask:
        result = (result > 0.5).astype(np.float32)
    return result


def _random_flip(
    image: torch.Tensor,
    mask: torch.Tensor,
    box: torch.Tensor,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply independent random horizontal and vertical flips."""
    if random.random() < 0.5:
        image = torch.flip(image, dims=[-1])
        mask = torch.flip(mask, dims=[-1])
        x1, y1, x2, y2 = box.unbind()
        box = torch.stack([W - x2, y1, W - x1, y2])

    if random.random() < 0.5:
        image = torch.flip(image, dims=[-2])
        mask = torch.flip(mask, dims=[-2])
        x1, y1, x2, y2 = box.unbind()
        box = torch.stack([x1, H - y2, x2, H - y1])

    return image, mask, box


# ---------------------------------------------------------------------------
# LoRA adapter for nn.Linear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adaptation (LoRA) wrapper around an existing ``nn.Linear``.

    The original weight matrix is frozen; only the low-rank matrices A and B
    are trained.  Output = original(x) + (x @ A.T @ B.T) * (alpha / rank).

    Args:
        linear: Original ``nn.Linear`` layer to adapt.
        rank:   LoRA rank (number of low-rank dimensions).
        alpha:  Scaling factor; effective scale = alpha / rank.
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 1.0) -> None:
        super().__init__()
        self.original = linear
        self.scale = alpha / rank

        for param in self.original.parameters():
            param.requires_grad_(False)

        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale


# ---------------------------------------------------------------------------
# Per-frame refinement head
# ---------------------------------------------------------------------------

class PerFrameRefinementHead(nn.Module):
    """Lightweight CNN that refines a coarse SAM2 mask.

    Input channels: image RGB (3) + coarse mask sigmoid (1) + box mask (1) = 5.
    Output: single-channel refined mask logit.

    This head is applied on every frame *before* temporal fusion, so per-frame
    improvements propagate directly into the SAM2 video pipeline.

    Args:
        in_channels: Number of input channels (default 5).
        features:    Number of intermediate feature channels (default 32).
    """

    def __init__(self, in_channels: int = 5, features: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, 1, kernel_size=1),
        )
        # Small init on final layer for stable training start
        nn.init.normal_(self.conv[-1].weight, std=0.01)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(
        self,
        image: torch.Tensor,
        coarse_mask: torch.Tensor,
        box_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image       : (B, 3, H, W) float32 in [0, 1]
            coarse_mask : (B, 1, H, W) coarse SAM2 sigmoid output
            box_mask    : (B, 1, H, W) binary box region

        Returns:
            (B, 1, H, W) refined mask logits
        """
        x = torch.cat([image, coarse_mask, box_mask], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bce_dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Combined Binary Cross-Entropy and Dice loss for binary segmentation.

    Args:
        pred_logits: (B, 1, H, W) raw logits.
        target:      (B, 1, H, W) binary ground-truth mask in {0, 1}.
        bce_weight:  Relative weight of BCE; (1 - bce_weight) weights Dice.
        smooth:      Laplace smoothing constant for Dice denominator.

    Returns:
        Scalar loss tensor.
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)

    pred_sigmoid = torch.sigmoid(pred_logits)
    intersection = (pred_sigmoid * target).sum(dim=(-2, -1))
    union = pred_sigmoid.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    dice = dice.mean()

    return bce_weight * bce + (1.0 - bce_weight) * dice


def boundary_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Edge-weighted BCE that penalises errors near mask boundaries.

    Detects ground-truth edges with a Sobel kernel and multiplies BCE weights
    by (1 + 5 * edge_magnitude) so the loss is amplified near boundaries.

    Args:
        pred_logits: (B, 1, H, W) raw logits.
        target:      (B, 1, H, W) binary ground-truth mask.

    Returns:
        Scalar loss tensor.
    """
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=target.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-2, -1)

    edges_x = F.conv2d(target, sobel_x, padding=1)
    edges_y = F.conv2d(target, sobel_y, padding=1)
    edges = (edges_x.pow(2) + edges_y.pow(2)).sqrt().clamp(0.0, 1.0)

    weights = 1.0 + 5.0 * edges
    return F.binary_cross_entropy_with_logits(pred_logits, target, weight=weights)


# ---------------------------------------------------------------------------
# LoRA injection helpers
# ---------------------------------------------------------------------------

def _inject_lora_into_module(
    module: nn.Module,
    rank: int,
    alpha: float,
    target_out_dims: set[int],
) -> int:
    """Recursively replace ``nn.Linear`` layers whose ``out_features`` match
    ``target_out_dims`` with ``LoRALinear`` wrappers.

    Returns the total number of replaced layers.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.out_features in target_out_dims:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += _inject_lora_into_module(child, rank, alpha, target_out_dims)
    return count


def build_trainable_sam2(
    model_id: str,
    lora_rank: int,
    lora_alpha: float,
    refinement_features: int,
    freeze_backbone: bool = True,
) -> tuple[Any, PerFrameRefinementHead]:
    """Load SAM2, inject LoRA adapters, and unfreeze small decoder components.

    Steps:
    1. Load ``SAM2ImagePredictor`` from HuggingFace hub.
    2. Optionally freeze all backbone parameters.
    3. Inject LoRA into linear layers of the image encoder's attention blocks.
    4. Unfreeze mask decoder and prompt encoder for full fine-tuning.
    5. Create and return a freshly initialised ``PerFrameRefinementHead``.

    Args:
        model_id:             HuggingFace model ID (e.g. ``facebook/sam2-hiera-large``).
        lora_rank:            LoRA matrix rank.
        lora_alpha:           LoRA scaling alpha.
        refinement_features:  Feature channels in refinement head.
        freeze_backbone:      If True, freeze all SAM2 weights before LoRA injection.

    Returns:
        ``(sam_model, refinement_head)`` tuple.
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    logger.info("Loading SAM2 from: %s", model_id)
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    sam_model = predictor.model

    if freeze_backbone:
        for param in sam_model.parameters():
            param.requires_grad_(False)
        logger.info("All SAM2 parameters frozen.")

    # Inject LoRA into image encoder attention linear layers.
    # Hiera-based SAM2 backbones use channels: 96, 192, 384, 768, 1024, 1280.
    encoder = getattr(sam_model, "image_encoder", None)
    lora_count = 0
    if encoder is not None:
        target_dims = {96, 192, 384, 768, 1024, 1280}
        lora_count = _inject_lora_into_module(encoder, lora_rank, lora_alpha, target_dims)
        logger.info("Injected LoRA into %d linear layers in image_encoder.", lora_count)
    else:
        logger.warning("image_encoder not found on SAM2 model; LoRA skipped.")

    # Unfreeze mask decoder (compact, ~3–4 M params)
    decoder = getattr(sam_model, "mask_decoder", None)
    if decoder is not None:
        for param in decoder.parameters():
            param.requires_grad_(True)
        logger.info("mask_decoder unfrozen.")

    # Unfreeze prompt encoder (tiny, ~6 K params)
    prompt_enc = getattr(sam_model, "prompt_encoder", None)
    if prompt_enc is not None:
        for param in prompt_enc.parameters():
            param.requires_grad_(True)
        logger.info("prompt_encoder unfrozen.")

    trainable = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam_model.parameters())
    logger.info(
        "SAM2 trainable params: %d / %d (%.1f %%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )

    refinement_head = PerFrameRefinementHead(
        in_channels=5, features=refinement_features
    )
    return sam_model, refinement_head


# ---------------------------------------------------------------------------
# SAM2 forward pass (training-compatible, with gradient tracking)
# ---------------------------------------------------------------------------

def sam2_forward_training(
    sam_model: Any,
    images: torch.Tensor,
    boxes_xyxy: torch.Tensor,
) -> torch.Tensor:
    """Run SAM2 image-prediction forward pass with gradient tracking.

    Replicates the internal steps of ``SAM2ImagePredictor.predict()`` but
    without ``torch.inference_mode`` so gradients propagate to LoRA adapters
    and the trainable decoder components.

    Coordinate convention for ``boxes_xyxy``: absolute pixel coords in the
    **input** image space (not normalised).  They are scaled to the model's
    internal image size before being passed to the prompt encoder.

    Args:
        sam_model:   ``SAM2Base`` instance (``predictor.model``).
        images:      (B, 3, H, W) float32 in [0, 1].
        boxes_xyxy:  (B, 4) float32 in absolute pixel XYXY coords.

    Returns:
        (B, 1, H, W) coarse mask logits upsampled to input resolution.
    """
    B, _, H, W = images.shape

    # SAM2's ImageEncoder applies its own ImageNet normalisation internally,
    # but it expects the input in the [0, 255] range.
    images_255 = images * 255.0

    # ---- Image encoding ----
    backbone_out = sam_model.image_encoder(images_255)

    # ---- Prepare features ----
    # SAM2Base._prepare_backbone_features converts the backbone dict into
    # flattened vision_feats (list of HW×B×C tensors) and feat_sizes.
    if hasattr(sam_model, "_prepare_backbone_features"):
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            sam_model._prepare_backbone_features(backbone_out)
        )
        # Reshape flattened feats back to (B, C, H_f, W_f) for the decoder.
        # vision_feats is ordered coarse→fine; last entry = most abstract.
        image_embed = (
            vision_feats[-1]
            .permute(1, 2, 0)
            .reshape(B, -1, feat_sizes[-1][0], feat_sizes[-1][1])
        )
        high_res_features = [
            vision_feats[i]
            .permute(1, 2, 0)
            .reshape(B, -1, feat_sizes[i][0], feat_sizes[i][1])
            for i in range(len(vision_feats) - 1)
        ]
    else:
        # Fallback for simplified or future API variants
        if isinstance(backbone_out, dict):
            feats = backbone_out.get("backbone_fpn", list(backbone_out.values()))
            image_embed = feats[-1] if isinstance(feats, list) else feats
        else:
            image_embed = backbone_out
        high_res_features = None

    # ---- Prompt encoding ----
    # SAM2 PromptEncoder expects boxes scaled to [0, image_size] where
    # image_size is the encoder's canonical resolution (usually 1024).
    model_image_size = getattr(sam_model, "image_size", 1024)
    scale_x = model_image_size / max(W, 1)
    scale_y = model_image_size / max(H, 1)
    scale = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=boxes_xyxy.device)
    boxes_scaled = boxes_xyxy * scale  # (B, 4) in model coordinates
    boxes_for_enc = boxes_scaled.unsqueeze(1)  # (B, 1, 4)

    sparse_emb, dense_emb = sam_model.prompt_encoder(
        points=None,
        boxes=boxes_for_enc,
        masks=None,
    )

    image_pe = sam_model.prompt_encoder.get_dense_pe()

    # ---- Mask decoding ----
    # Try the SAM2 decoder with high_res_features (SAM2 ≥ 1.0); fall back to
    # the simpler SAM1-compatible signature.
    decode_kwargs: dict[str, Any] = dict(
        image_embeddings=image_embed,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
    )
    if high_res_features is not None:
        decode_kwargs["high_res_features"] = high_res_features

    try:
        low_res_masks, _iou, _, _ = sam_model.mask_decoder(**decode_kwargs)
    except TypeError:
        # Retry without unsupported keyword args
        for optional_key in ("high_res_features", "repeat_image"):
            decode_kwargs.pop(optional_key, None)
        low_res_masks, _iou, _, _ = sam_model.mask_decoder(**decode_kwargs)

    # Upsample to the original input resolution
    return F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Box-mask utility
# ---------------------------------------------------------------------------

def boxes_to_mask(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert XYXY box tensors to binary spatial masks.

    Args:
        boxes: (B, 4) float32 XYXY in absolute pixel coords.
        H, W:  Output spatial dimensions.

    Returns:
        (B, 1, H, W) float32 binary mask where the box interior is 1.
    """
    B = boxes.shape[0]
    out = torch.zeros(B, 1, H, W, device=boxes.device, dtype=torch.float32)
    for i in range(B):
        x1, y1, x2, y2 = boxes[i].long()
        x1 = x1.clamp(0, W - 1)
        x2 = x2.clamp(x1 + 1, W)
        y1 = y1.clamp(0, H - 1)
        y2 = y2.clamp(y1 + 1, H)
        out[i, 0, y1:y2, x1:x2] = 1.0
    return out


# ---------------------------------------------------------------------------
# PyTorch Lightning module
# ---------------------------------------------------------------------------

class SAM2MacropinosomeModule:
    """PyTorch Lightning module for SAM2 adapter fine-tuning.

    Trains with a combined loss on:
    - Refined mask (primary): BCE + Dice
    - Coarse SAM2 mask (auxiliary, weight 0.3): BCE + Dice
    - Boundary/edge loss (optional, applied to refined mask)

    Three separate Adam parameter groups allow independent LR for LoRA
    adapters, the SAM2 decoder components, and the refinement head.
    """

    def __new__(  # type: ignore[return]
        cls,
        sam_model: Any,
        refinement_head: PerFrameRefinementHead,
        lr_adapters: float,
        lr_decoder: float,
        lr_refinement: float,
        weight_decay: float,
        bce_weight: float,
        use_boundary_loss: bool,
        boundary_weight: float,
        max_epochs: int,
    ) -> "SAM2MacropinosomeModule":
        try:
            import pytorch_lightning as pl
        except ImportError as exc:
            raise ImportError(
                "pytorch_lightning is required. Install via the deep-learning "
                "dependency group: uv add --group deep-learning pytorch-lightning"
            ) from exc

        class _Module(pl.LightningModule):
            def __init__(
                self,
                sam_model: Any,
                refinement_head: PerFrameRefinementHead,
                lr_adapters: float,
                lr_decoder: float,
                lr_refinement: float,
                weight_decay: float,
                bce_weight: float,
                use_boundary_loss: bool,
                boundary_weight: float,
                max_epochs: int,
            ) -> None:
                super().__init__()
                self.save_hyperparameters(ignore=["sam_model", "refinement_head"])
                self.sam_model = sam_model
                self.refinement_head = refinement_head

            def _shared_step(
                self, batch: tuple[torch.Tensor, ...], stage: str
            ) -> torch.Tensor:
                images, boxes, gt_masks = batch
                B, _, H, W = images.shape

                coarse_masks = sam2_forward_training(self.sam_model, images, boxes)
                coarse_mask = coarse_masks[:, :1]  # best candidate only

                box_masks = boxes_to_mask(boxes, H, W)
                refined_logits = self.refinement_head(
                    images, torch.sigmoid(coarse_mask), box_masks
                )

                # Primary loss
                hp = self.hparams
                loss = bce_dice_loss(refined_logits, gt_masks, bce_weight=hp.bce_weight)

                # Auxiliary loss on coarse SAM2 output
                loss = loss + 0.3 * bce_dice_loss(
                    coarse_mask, gt_masks, bce_weight=hp.bce_weight
                )

                # Optional boundary loss
                if hp.use_boundary_loss:
                    loss = loss + hp.boundary_weight * boundary_loss(
                        refined_logits, gt_masks
                    )

                with torch.no_grad():
                    pred_binary = (torch.sigmoid(refined_logits) > 0.5).float()
                    inter = (pred_binary * gt_masks).sum(dim=(-3, -2, -1))
                    union_mask = (pred_binary + gt_masks).clamp(0, 1).sum(dim=(-3, -2, -1))
                    iou = (inter / (union_mask + 1e-8)).mean()

                on_step = stage == "train"
                self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=on_step)
                self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True)
                return loss

            def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
                return self._shared_step(batch, "train")

            def validation_step(self, batch: tuple, batch_idx: int) -> None:
                self._shared_step(batch, "val")

            def configure_optimizers(self):
                hp = self.hparams
                lora_params = [
                    p
                    for n, p in self.sam_model.named_parameters()
                    if p.requires_grad and "lora_" in n
                ]
                decoder_params = [
                    p
                    for n, p in self.sam_model.named_parameters()
                    if p.requires_grad and ("mask_decoder" in n or "prompt_encoder" in n)
                    and "lora_" not in n
                ]
                refinement_params = list(self.refinement_head.parameters())

                param_groups = []
                if lora_params:
                    param_groups.append(
                        {"params": lora_params, "lr": hp.lr_adapters, "name": "lora"}
                    )
                if decoder_params:
                    param_groups.append(
                        {"params": decoder_params, "lr": hp.lr_decoder, "name": "decoder"}
                    )
                param_groups.append(
                    {
                        "params": refinement_params,
                        "lr": hp.lr_refinement,
                        "name": "refinement_head",
                    }
                )

                optimizer = torch.optim.AdamW(
                    param_groups, weight_decay=hp.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=hp.max_epochs, eta_min=1e-7
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                    },
                }

        return _Module(
            sam_model,
            refinement_head,
            lr_adapters,
            lr_decoder,
            lr_refinement,
            weight_decay,
            bce_weight,
            use_boundary_loss,
            boundary_weight,
            max_epochs,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901  (acceptable complexity for CLI entry point)
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune SAM2 adapters for macropinosome segmentation.\n\n"
            "Loads paired image patches and SAM2 binary masks from the output of\n"
            "deep_learning_02_make_training_set_sam2.py, injects LoRA adapters into\n"
            "the SAM2 image encoder, fine-tunes the compact mask decoder and prompt\n"
            "encoder, and trains a per-frame refinement head.\n\n"
            "Note: --input-folder and --output-folder are used instead of\n"
            "--input-search-pattern because this script reads a fixed folder\n"
            "structure (training_set/input/ and training_set/target_sam2/)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Fine-tune SAM2 adapters (default settings)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_macropinosome_training.py'
  - --input-folder: '%YAML%/deep_learning_output'

- name: Fine-tune SAM2 adapters (custom hyperparameters)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_macropinosome_training.py'
  - --input-folder: '%YAML%/deep_learning_output'
  - --output-folder: '%YAML%/sam2_adapted'
  - --model-id: facebook/sam2-hiera-large
  - --lora-rank: 16
  - --lora-alpha: 1.0
  - --max-epochs: 80
  - --batch-size: 8
  - --lr-adapters: 1e-4
  - --lr-decoder: 1e-4
  - --lr-refinement: 5e-4
  - --val-split: 0.15
  - --patch-size: 256
  - --boundary-loss
  - --augment

- name: Pause to inspect TensorBoard logs
  type: pause
  message: 'Run: tensorboard --logdir <output-folder>/sam2_adapter_logs'

- name: Stop intentionally
  type: stop
  message: 'Pipeline stopped intentionally.'

- name: Force reprocessing for later segments
  type: force
  message: 'Reprocessing all subsequent steps.'
""",
    )

    # ---- Data arguments ----
    parser.add_argument(
        "--input-folder",
        required=True,
        help=(
            "Root output folder from deep_learning_02_make_training_set_sam2.py. "
            "Must contain training_set/input/ and training_set/target_sam2/."
        ),
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help=(
            "Folder for checkpoints and TensorBoard logs. "
            "Defaults to <input-folder>/sam2_adapted/."
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=0,
        help=(
            "Resize all patches to this square size before passing to SAM2. "
            "0 = keep native patch size (default). "
            "Use a power-of-two (e.g. 128, 256) for consistent batching."
        ),
    )

    # ---- SAM2 model ----
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam2-hiera-large",
        help="HuggingFace model ID for SAM2 (default: facebook/sam2-hiera-large).",
    )

    # ---- Adapter settings ----
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help=(
            "LoRA rank injected into image encoder attention linear layers "
            "(default: 16). Higher = more capacity, more parameters."
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=1.0,
        help="LoRA scaling alpha; effective scale = alpha / rank (default: 1.0).",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help=(
            "Do NOT freeze SAM2 backbone before LoRA injection. "
            "Warning: trains all SAM2 parameters — requires much more memory."
        ),
    )
    parser.add_argument(
        "--refinement-features",
        type=int,
        default=32,
        help="Feature channels in the per-frame refinement CNN head (default: 32).",
    )

    # ---- Loss settings ----
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help=(
            "Relative weight of BCE in the combined BCE + Dice loss "
            "(default: 0.5; set to 1.0 for pure BCE)."
        ),
    )
    parser.add_argument(
        "--boundary-loss",
        action="store_true",
        help="Add an edge-weighted boundary loss term to the refined mask loss.",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.1,
        help="Weight of the boundary loss term when --boundary-loss is set (default: 0.1).",
    )

    # ---- Optimisation ----
    parser.add_argument(
        "--lr-adapters",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA adapter weights (default: 1e-4).",
    )
    parser.add_argument(
        "--lr-decoder",
        type=float,
        default=1e-4,
        help="Learning rate for mask decoder and prompt encoder (default: 1e-4).",
    )
    parser.add_argument(
        "--lr-refinement",
        type=float,
        default=5e-4,
        help="Learning rate for the per-frame refinement head (default: 5e-4).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
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
        help="Fraction of data held out for validation (default: 0.15).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience in epochs (default: 20; 0 = disable).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable random horizontal/vertical flip augmentation.",
    )
    parser.add_argument(
        "--box-margin",
        type=int,
        default=4,
        help="Extra pixels added around the derived bounding box (default: 4).",
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

    # ---- Resolve paths ----
    input_folder = Path(args.input_folder)
    input_dir = input_folder / "training_set" / "input"
    mask_dir = input_folder / "training_set" / "target_sam2"

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        raise SystemExit(1)
    if not mask_dir.is_dir():
        logger.error("Mask directory not found: %s", mask_dir)
        raise SystemExit(1)

    output_folder = Path(args.output_folder) if args.output_folder else (
        input_folder / "sam2_adapted"
    )
    ckpt_dir = output_folder / "checkpoints"
    log_dir = output_folder / "sam2_adapter_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Input:   %s", input_dir)
    logger.info("Masks:   %s", mask_dir)
    logger.info("Output:  %s", output_folder)

    # ---- Reproducibility ----
    import pytorch_lightning as pl

    pl.seed_everything(args.seed, workers=True)

    # ---- Dataset ----
    full_dataset = MacropinosomeDataset(
        input_dir=input_dir,
        mask_dir=mask_dir,
        patch_size=args.patch_size,
        augment=args.augment,
        box_margin=args.box_margin,
    )

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    if n_train < 1:
        logger.error(
            "Not enough samples for training (total=%d, val=%d).", n_total, n_val
        )
        raise SystemExit(1)

    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info("Train: %d  |  Val: %d", n_train, n_val)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(n_train >= args.batch_size),
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    sam_model, refinement_head = build_trainable_sam2(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        refinement_features=args.refinement_features,
        freeze_backbone=not args.no_freeze_backbone,
    )

    lightning_module = SAM2MacropinosomeModule(
        sam_model=sam_model,
        refinement_head=refinement_head,
        lr_adapters=args.lr_adapters,
        lr_decoder=args.lr_decoder,
        lr_refinement=args.lr_refinement,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        use_boundary_loss=args.boundary_loss,
        boundary_weight=args.boundary_weight,
        max_epochs=args.max_epochs,
    )

    # ---- Callbacks ----
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}-val_iou{val_iou:.4f}",
            monitor="val_iou",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                mode="min",
                verbose=True,
            )
        )

    tb_logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name="sam2_adapters",
    )

    # ---- Trainer ----
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info("Training on: %s", accelerator.upper())

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else "32",
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=max(1, n_train // args.batch_size // 5),
        enable_progress_bar=True,
    )

    logger.info(
        "Starting training  |  epochs=%d  |  batch=%d  |  LoRA rank=%d",
        args.max_epochs,
        args.batch_size,
        args.lora_rank,
    )
    trainer.fit(lightning_module, train_loader, val_loader)

    # ---- Save final adapter weights ----
    best_ckpt = callbacks[0].best_model_path
    logger.info("Best checkpoint: %s", best_ckpt)

    # Export only the trainable parameters for compact sharing
    trainable_state = {
        k: v
        for k, v in lightning_module.state_dict().items()
        if v.requires_grad or "lora_" in k or "refinement_head" in k
    }
    adapter_path = output_folder / "sam2_adapter_weights.pt"
    torch.save(
        {
            "model_id": args.model_id,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "refinement_features": args.refinement_features,
            "trainable_state_dict": trainable_state,
        },
        adapter_path,
    )
    logger.info("Adapter weights saved to: %s", adapter_path)


if __name__ == "__main__":
    main()
