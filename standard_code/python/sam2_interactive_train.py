"""
sam2_interactive_train.py
--------------------------
All-in-one interactive napari tool: annotate boxes, see SAM2's current
prediction, accept the algorithmic target mask, train on all accumulated
samples, then immediately see whether the model improved — all in one session.

Workflow (press-by-press)
--------------------------
1. All patches from matching input files open as a (N, H, W) napari stack.
2. Navigate to a patch containing a macropinosome (N-axis slider).
3. Draw a rectangle in the 'Boxes' layer.
4. Processing runs automatically after the rectangle is completed:
      Blue  = SAM2's CURRENT prediction (before this sample is used).
      Green = algorithmic target mask (BIPHUB ImageJ port).
5. Press A (accept):
      - Saves crop + algorithmic mask to training_set/input and target_sam2/.
      - Runs a quick fine-tuning pass on ALL accumulated samples.
      - Cyan = SAM2's UPDATED prediction after training — see if it improved.
6. Press D (discard): clears all overlays and the drawn box; redraw.

Adapter weights are saved to <output-folder>/sam2_adapter_weights.pt after
every training round.  On restart the weights are loaded automatically so
you can continue improving the same model across multiple sessions.

Algorithmic segmentation (Python port of the BIPHUB ImageJ macro)
------------------------------------------------------------------
  1. Grayscale fill-holes via morphological reconstruction from border seeds.
  2. Sample the pixel value at the box centre after filling.
  3. Threshold: keep pixels in [centre_val − tol, centre_val + tol].
  4. Binary fill-holes.
  5. 4-connected component labelling; keep the component at the box centre.

Output folder structure
-----------------------
<output-folder>/
    training_set/
        input/          float32 image crops (.tif)
        target_sam2/    float32 binary algorithmic masks (.tif, 0.0 or 1.0)
    sam2_adapter_weights.pt   LoRA + decoder + refinement checkpoint

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Algorithmic segmentation (Python port of the BIPHUB ImageJ macro)
# ---------------------------------------------------------------------------

def segment_algorithmic(
    crop_2d: np.ndarray,
    center_y: int,
    center_x: int,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Segment a macropinosome using the BIPHUB macropinosome algorithm.

    Args:
        crop_2d:   (H, W) float or integer image array.
        center_y:  Row index of the seed point.
        center_x:  Column index of the seed point.
        tolerance: ±tolerance around the centre pixel value for the threshold.

    Returns:
        (H, W) float32 binary mask; 1.0 inside the macropinosome.
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label
    from skimage.morphology import reconstruction

    crop_f = crop_2d.astype(np.float32)
    H, W = crop_f.shape

    seed = crop_f.copy()
    seed[1:-1, 1:-1] = crop_f.max()
    filled = reconstruction(seed, crop_f, method="erosion").astype(np.float32)

    cy = int(np.clip(center_y, 0, H - 1))
    cx = int(np.clip(center_x, 0, W - 1))
    center_val = float(filled[cy, cx])

    binary = (filled >= center_val - tolerance) & (filled <= center_val + tolerance)
    binary = binary_fill_holes(binary)

    labeled = label(binary.astype(np.uint8), connectivity=1)
    comp_id = int(labeled[cy, cx])
    if comp_id == 0:
        logger.warning(
            "No component at seed (%d, %d); returning empty mask.", cy, cx
        )
        return np.zeros((H, W), dtype=np.float32)

    return (labeled == comp_id).astype(np.float32)


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around an existing nn.Linear.

    Output = original(x) + lora_B(lora_A(x)) * (alpha / rank).
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

    Input channels: image RGB (3) + coarse sigmoid (1) + box mask (1) = 5.
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
        nn.init.normal_(self.conv[-1].weight, std=0.01)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(
        self, image: torch.Tensor, coarse: torch.Tensor, box_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(torch.cat([image, coarse, box_mask], dim=1))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bce_dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Combined BCE + Dice loss."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred_sigmoid = torch.sigmoid(pred_logits)
    inter = (pred_sigmoid * target).sum(dim=(-2, -1))
    union = pred_sigmoid.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return bce_weight * bce + (1.0 - bce_weight) * dice.mean()


# ---------------------------------------------------------------------------
# LoRA injection and model setup
# ---------------------------------------------------------------------------

def _inject_lora(module: nn.Module, rank: int, alpha: float, dims: set[int]) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.out_features in dims:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += _inject_lora(child, rank, alpha, dims)
    return count


def build_trainable_sam2(
    model_id: str,
    lora_rank: int,
    lora_alpha: float,
    refinement_features: int,
) -> tuple[Any, PerFrameRefinementHead, Any]:
    """Load SAM2, inject LoRA, unfreeze decoder.

    Returns:
        (sam_model, refinement_head, predictor)
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    logger.info("Loading SAM2: %s", model_id)
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    sam_model = predictor.model

    # Freeze all backbone
    for param in sam_model.parameters():
        param.requires_grad_(False)

    # Inject LoRA into Hiera attention linears
    enc = getattr(sam_model, "image_encoder", None)
    if enc is not None:
        n = _inject_lora(enc, lora_rank, lora_alpha, {144, 288, 432, 576, 864, 1152, 1728, 2304})
        logger.info("LoRA injected into %d linear layers.", n)

    # Unfreeze mask decoder + prompt encoder (SAM2 uses "sam_" prefix)
    for attr in ("sam_mask_decoder", "sam_prompt_encoder", "mask_decoder", "prompt_encoder"):
        m = getattr(sam_model, attr, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(True)
            logger.info("%s unfrozen.", attr)

    trainable = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam_model.parameters())
    logger.info("Trainable: %d / %d (%.1f%%)", trainable, total, 100 * trainable / max(total, 1))

    head = PerFrameRefinementHead(in_channels=5, features=refinement_features)
    return sam_model, head, predictor


# ---------------------------------------------------------------------------
# SAM2 forward pass (gradient-tracked, for training)
# ---------------------------------------------------------------------------

def sam2_forward_training(
    sam_model: Any, images: torch.Tensor, boxes_xyxy: torch.Tensor
) -> torch.Tensor:
    """Run SAM2 forward pass with gradient tracking.

    Args:
        images:     (B, 3, H, W) float32 in [0, 1].
        boxes_xyxy: (B, 4) absolute pixel XYXY coords.

    Returns:
        (B, 1, H, W) coarse mask logits at input resolution.
    """
    B, _, H, W = images.shape
    images_255 = images * 255.0

    # SAM2 uses "sam_" prefixed attributes; fall back to unprefixed for other variants
    _prompt_enc = getattr(sam_model, "sam_prompt_encoder", None) or getattr(sam_model, "prompt_encoder")
    _mask_dec   = getattr(sam_model, "sam_mask_decoder",   None) or getattr(sam_model, "mask_decoder")

    # SAM2 expects its native resolution (1024×1024); resize before encoding
    model_size = getattr(sam_model, "image_size", 1024)
    if H != model_size or W != model_size:
        images_enc = F.interpolate(
            images_255, size=(model_size, model_size), mode="bilinear", align_corners=False
        )
    else:
        images_enc = images_255

    # Use forward_image (includes conv_s0/conv_s1 projections for high-res features)
    # so that LoRA gradients flow through image_encoder and the projections.
    backbone_out = sam_model.forward_image(images_enc)

    # Reshape features exactly as SAM2ImagePredictor.set_image does
    _, vision_feats, _, feat_sizes = sam_model._prepare_backbone_features(backbone_out)

    # Add no_mem_embed to the lowest-res feature (same as the predictor)
    if getattr(sam_model, "directly_add_no_mem_embed", False):
        vision_feats[-1] = vision_feats[-1] + sam_model.no_mem_embed

    # Mirror SAM2ImagePredictor.set_image exactly:
    #   feats = reversed list → feats[-1] = finest = image_embed
    feats = [
        feat.permute(1, 2, 0).reshape(B, -1, fsz[0], fsz[1])
        for feat, fsz in zip(vision_feats[::-1], feat_sizes[::-1])
    ]
    # feats[0]  = coarsest (B, 256,  64, 64) → high_res[0] (feat_s0 in decoder)
    # feats[1]  = medium   (B,  64, 128,128) → high_res[1] (feat_s1 in decoder)
    # feats[-1] = finest   (B,  32, 256,256) → image_embed (matches image_pe spatial dims)
    image_embed = feats[-1]
    high_res    = feats[:-1]

    logger.debug(
        "sam2_forward_training: image_embed=%s  high_res=%s  boxes=%s",
        tuple(image_embed.shape),
        [tuple(f.shape) for f in high_res],
        boxes_xyxy.tolist(),
    )

    scale = torch.tensor(
        [model_size / W, model_size / H, model_size / W, model_size / H],
        device=boxes_xyxy.device,
    )
    sparse_emb, dense_emb = _prompt_enc(
        points=None, boxes=(boxes_xyxy * scale).unsqueeze(1), masks=None
    )
    image_pe = _prompt_enc.get_dense_pe()

    kwargs: dict[str, Any] = dict(
        image_embeddings=image_embed,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
    )
    if high_res:
        kwargs["high_res_features"] = high_res

    try:
        low_res, *_ = _mask_dec(**kwargs)
    except TypeError:
        for k in ("high_res_features", "repeat_image"):
            kwargs.pop(k, None)
        low_res, *_ = _mask_dec(**kwargs)

    return F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Box-mask utility
# ---------------------------------------------------------------------------

def boxes_to_mask(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert (B, 4) XYXY box tensors to (B, 1, H, W) binary masks."""
    B = boxes.shape[0]
    out = torch.zeros(B, 1, H, W, device=boxes.device, dtype=torch.float32)
    for i in range(B):
        x1, y1, x2, y2 = boxes[i].long()
        x1, x2 = x1.clamp(0, W - 1), x2.clamp(0, W)
        y1, y2 = y1.clamp(0, H - 1), y2.clamp(0, H)
        out[i, 0, y1:y2, x1:x2] = 1.0
    return out


# ---------------------------------------------------------------------------
# Normalisation helper (for inference)
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 0.1), np.percentile(arr, 99.9)
    out = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# SAM2 inference  (before / after training)
# ---------------------------------------------------------------------------

def _predict_sam2(
    predictor: Any,
    refinement_head: PerFrameRefinementHead,
    crop_2d: np.ndarray,
    box_xyxy_in_crop: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run SAM2 + refinement head inference.  Returns (H, W) float32 mask in [0, 1]."""
    H, W = crop_2d.shape
    img_u8 = _to_uint8(crop_2d)
    img_rgb = np.stack([img_u8] * 3, axis=-1)  # (H, W, 3) uint8 — standard predictor API

    x1, y1, x2, y2 = box_xyxy_in_crop.astype(np.float32)
    box_np = np.array([x1, y1, x2, y2])

    predictor.model.eval()
    refinement_head.eval()
    with torch.inference_mode():
        # Use the predictor API directly — no manual forward pass needed
        predictor.set_image(img_rgb)
        masks, _scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np,
            multimask_output=False,
        )
        # Return the raw SAM2 coarse mask.  The refinement head is untrained at
        # the start so applying it just yields ~0.5 everywhere (uninformative).
        # After training rounds the caller can switch to using the refined mask.
        return masks[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Patch loading (mirrors deep_learning_01_make_training_set.py)
# ---------------------------------------------------------------------------

def _load_patches(
    files: list[str],
    patch_size: int,
    channel: int,
    z_index: int,
    max_patches: int | None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """Load one frame per file and extract a non-overlapping patch grid.

    Returns:
        patches_arr : (N, patch_size, patch_size) float32.
        patch_meta  : list of (file_idx, r0, c0, r1, c1).
    """
    frames: list[np.ndarray] = []
    for fpath in files:
        img = rp.load_tczyx_image(fpath)
        ch = int(np.clip(channel, 0, img.dims.C - 1))
        zi = int(np.clip(z_index if z_index >= 0 else img.dims.Z // 2, 0, img.dims.Z - 1))
        t_mid = img.dims.T // 2
        block = img.get_image_data("ZYX", T=t_mid, C=ch)
        frame = block[zi] if block.shape[0] > 1 else block[0]
        frames.append(frame.astype(np.float32))
        logger.info("Loaded %s  shape=%s  ch=%d  z=%d", Path(fpath).name, frame.shape, ch, zi)

    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        p = np.zeros((max_h, max_w), dtype=np.float32)
        p[: f.shape[0], : f.shape[1]] = f
        padded.append(p)

    ps = patch_size
    all_patches: list[np.ndarray] = []
    patch_meta: list[tuple[int, int, int, int, int]] = []

    for file_idx, frame in enumerate(padded):
        H, W = frame.shape
        for r0 in range(0, H, ps):
            for c0 in range(0, W, ps):
                r1, c1 = min(r0 + ps, H), min(c0 + ps, W)
                patch = np.zeros((ps, ps), dtype=np.float32)
                patch[: r1 - r0, : c1 - c0] = frame[r0:r1, c0:c1]
                all_patches.append(patch)
                patch_meta.append((file_idx, r0, c0, r1, c1))
                if max_patches is not None and len(all_patches) >= max_patches:
                    break
            if max_patches is not None and len(all_patches) >= max_patches:
                break

    patches_arr = np.stack(all_patches).astype(np.float32)
    logger.info("Patches: %d  shape=%s  from %d file(s)", len(all_patches), patches_arr.shape, len(files))
    return patches_arr, patch_meta


# ---------------------------------------------------------------------------
# Quick training pass on all accumulated samples
# ---------------------------------------------------------------------------

def _train_round(
    sam_model: Any,
    refinement_head: PerFrameRefinementHead,
    input_dir: Path,
    target_dir: Path,
    lr: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> float:
    """Fine-tune on all saved samples.  Returns the final mean loss."""
    from torch.utils.data import DataLoader, Dataset

    # ---- Inline dataset (no separate class needed) ------------------------
    input_paths = sorted(input_dir.glob("*.tif"), key=lambda p: int(p.stem))
    target_paths = sorted(target_dir.glob("*.tif"), key=lambda p: int(p.stem))
    # Match by stem
    input_by_stem = {p.stem: p for p in input_paths}
    target_by_stem = {p.stem: p for p in target_paths}
    common = sorted(input_by_stem.keys() & target_by_stem.keys(), key=int)

    if not common:
        logger.warning("No training pairs found; skipping training.")
        return float("nan")

    logger.info("[TRAIN] Found %d paired sample(s).", len(common))

    class _PairDataset(Dataset):
        def __init__(self) -> None:
            self.pairs = [(input_by_stem[s], target_by_stem[s]) for s in common]

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            img = tifffile.imread(self.pairs[i][0]).astype(np.float32)
            msk = tifffile.imread(self.pairs[i][1]).astype(np.float32)
            while img.ndim > 2:
                img = img[0]
            while msk.ndim > 2:
                msk = msk[0]

            lo = float(np.percentile(img, 0.1))
            hi = float(np.percentile(img, 99.9))
            img = np.clip((img - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
            msk = (msk > 0.5).astype(np.float32)
            H, W = img.shape

            nz_r, nz_c = np.where(msk > 0.5)
            m = 4
            if len(nz_r):
                y1 = max(0, int(nz_r.min()) - m)
                y2 = min(H, int(nz_r.max()) + m + 1)
                x1 = max(0, int(nz_c.min()) - m)
                x2 = min(W, int(nz_c.max()) + m + 1)
            else:
                y1, x1, y2, x2 = 0, 0, H, W

            # Augmentation: random flips
            img_t = torch.from_numpy(np.stack([img] * 3))   # (3, H, W)
            msk_t = torch.from_numpy(msk).unsqueeze(0)       # (1, H, W)
            box_t = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[-1])
                msk_t = torch.flip(msk_t, dims=[-1])
                bx1, by1, bx2, by2 = box_t.unbind()
                box_t = torch.stack([W - bx2, by1, W - bx1, by2])

            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[-2])
                msk_t = torch.flip(msk_t, dims=[-2])
                bx1, by1, bx2, by2 = box_t.unbind()
                box_t = torch.stack([bx1, H - by2, bx2, H - by1])

            return img_t, box_t, msk_t

    ds = _PairDataset()
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # ---- Parameter groups -------------------------------------------------
    lora_params = [p for n, p in sam_model.named_parameters() if p.requires_grad and "lora_" in n]
    decoder_params = [
        p for n, p in sam_model.named_parameters()
        if p.requires_grad and ("mask_decoder" in n or "prompt_encoder" in n) and "lora_" not in n
    ]
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr})
    param_groups.append({"params": list(refinement_head.parameters()), "lr": lr * 5})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    sam_model.train()
    refinement_head.train()

    final_loss = float("nan")
    logger.info(
        "[TRAIN] Starting training round: epochs=%d batch_size=%d lr=%.3e",
        epochs,
        batch_size,
        lr,
    )
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for images, boxes, gt_masks in loader:
            images = images.to(device)
            boxes = boxes.to(device)
            gt_masks = gt_masks.to(device)
            B, _, H, W = images.shape

            optimizer.zero_grad()
            coarse = sam2_forward_training(sam_model, images, boxes)[:, :1]
            box_mask = boxes_to_mask(boxes, H, W)
            refined = refinement_head(images, torch.sigmoid(coarse), box_mask)

            loss = bce_dice_loss(refined, gt_masks) + 0.3 * bce_dice_loss(coarse, gt_masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            "[TRAIN] Epoch %d/%d  loss=%.4f  n_samples=%d",
            epoch + 1,
            epochs,
            final_loss,
            len(ds),
        )

    sam_model.eval()
    refinement_head.eval()
    logger.info("[TRAIN] Training round completed. final_loss=%.4f", final_loss)
    return final_loss


# ---------------------------------------------------------------------------
# Main interactive session
# ---------------------------------------------------------------------------

def run_interactive_train(
    files: list[str],
    output_folder: Path,
    channel: int,
    z_index: int,
    patch_size: int,
    max_patches: int | None,
    margin: int,
    tolerance: float,
    model_id: str,
    lora_rank: int,
    lora_alpha: float,
    refinement_features: int,
    lr: float,
    batch_size: int,
    train_epochs: int,
) -> None:
    """Open napari session with annotation + on-accept training loop."""
    import napari

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Build / restore model --------------------------------------------
    sam_model, refinement_head, _predictor = build_trainable_sam2(
        model_id, lora_rank, lora_alpha, refinement_features
    )
    sam_model = sam_model.to(device)
    refinement_head = refinement_head.to(device)

    weights_path = output_folder / "sam2_adapter_weights.pt"
    if weights_path.exists():
        logger.info("Restoring adapter weights from %s", weights_path)
        checkpoint = torch.load(str(weights_path), map_location=device)
        state = checkpoint.get("trainable_state_dict", checkpoint)
        missing, unexpected = sam_model.load_state_dict(state.get("sam_model", {}), strict=False)
        if missing:
            logger.debug("Missing keys: %s", missing[:5])
        refinement_head.load_state_dict(state.get("refinement_head", {}), strict=False)
        logger.info("Adapter weights loaded.")
    else:
        logger.info("No existing weights found — starting from base SAM2.")

    # ---- Load patches -------------------------------------------------------
    patches_arr, patch_meta = _load_patches(files, patch_size, channel, z_index, max_patches)
    N, ps_h, ps_w = patches_arr.shape

    # ---- Output directories ------------------------------------------------
    input_dir = output_folder / "training_set" / "input"
    target_dir = output_folder / "training_set" / "target_sam2"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted([int(p.stem) for p in input_dir.glob("*.tif") if p.stem.isdigit()])
    sample_counter = [existing[-1] + 1 if existing else 0]

    # ---- Mutable session state (closure-shared dict) -----------------------
    pending: dict = {
        "crop": None,
        "box_xyxy_in_crop": None,
        "algo_mask": None,
        "crop_origin": None,   # (patch_idx, ay1, ax1) for overlay placement
    }

    # ---- Napari viewer -----------------------------------------------------
    file_names = ", ".join(Path(f).name for f in files[:3])
    if len(files) > 3:
        file_names += f" … (+{len(files) - 3} more)"

    viewer = napari.Viewer(
        title=f"SAM2 Interactive Train — {file_names}  [{N} patches]"
    )
    viewer.add_image(patches_arr, name="Patches", colormap="gray")

    shapes_layer = viewer.add_shapes(
        name="Boxes",
        ndim=3,
        shape_type="rectangle",
        edge_color="yellow",
        face_color="transparent",
        edge_width=2,
    )

    # (N, ps_h, ps_w) overlay arrays
    sam2_before = np.zeros((N, ps_h, ps_w), dtype=np.float32)
    algo_overlay = np.zeros((N, ps_h, ps_w), dtype=np.float32)
    sam2_after  = np.zeros((N, ps_h, ps_w), dtype=np.float32)

    before_layer = viewer.add_image(
        sam2_before,
        name="SAM2 before training  [blue]",
        colormap="blue",
        opacity=0.45,
        visible=False,
        blending="additive",
    )
    algo_layer = viewer.add_image(
        algo_overlay,
        name="Algorithmic target  [green — A to save]",
        colormap="green",
        opacity=0.60,
        visible=False,
        blending="additive",
    )
    after_layer = viewer.add_image(
        sam2_after,
        name="SAM2 after training  [cyan]",
        colormap="cyan",
        opacity=0.55,
        visible=False,
        blending="additive",
    )

    viewer.status = (
        f"{N} patches loaded. Navigate the N-slider, draw a rectangle "
        "in 'Boxes' - processing starts automatically. Click the canvas, then press A to train. D = discard."
    )

    # Suppress auto-process while clearing shapes
    _auto_process_enabled = [True]
    _prev_shape_count = [0]

    # ---- Helper: read last drawn box (3-D shape coords: [N, Y, X]) ---------
    def _last_box_nyx() -> tuple[int, int, int, int, int] | None:
        if not shapes_layer.data:
            return None
        pts = np.array(shapes_layer.data[-1])
        if pts.shape[1] == 3:
            # 3D coords stored as [N, Y, X]
            patch_idx = int(np.clip(int(round(float(pts[:, 0].mean()))), 0, N - 1))
            y1, x1 = int(pts[:, 1].min()), int(pts[:, 2].min())
            y2, x2 = int(pts[:, 1].max()), int(pts[:, 2].max())
        else:
            # 2D coords stored as [Y, X] — use current slider position for N
            patch_idx = int(np.clip(viewer.dims.current_step[0], 0, N - 1))
            y1, x1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            y2, x2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        return patch_idx, y1, x1, y2, x2

    # ---- Auto-process when a new rectangle is completed -------------------
    def on_process() -> None:
        result = _last_box_nyx()
        if result is None:
            viewer.status = "No box drawn. Draw a rectangle in the 'Boxes' layer first."
            return

        patch_idx, y1, x1, y2, x2 = result
        patch = patches_arr[patch_idx]

        ay1 = max(0, y1 - margin)
        ax1 = max(0, x1 - margin)
        ay2 = min(ps_h, y2 + margin)
        ax2 = min(ps_w, x2 + margin)
        crop = patch[ay1:ay2, ax1:ax2].copy()
        crop_h, crop_w = crop.shape

        cy = (y1 + y2) // 2 - ay1
        cx = (x1 + x2) // 2 - ax1

        # Algorithmic mask
        viewer.status = "Running algorithmic segmentation…"
        try:
            algo_mask = segment_algorithmic(crop, cy, cx, tolerance=tolerance)
        except Exception as exc:
            logger.exception("Algorithmic segmentation failed: %s", exc)
            viewer.status = f"Segmentation error: {exc}"
            return

        # SAM2 current prediction
        box_in_crop = np.array([x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1], dtype=np.float32)
        viewer.status = "Running SAM2 (current)…"
        try:
            sam2_before_mask = _predict_sam2(
                _predictor, refinement_head, crop, box_in_crop, device
            )
        except Exception as exc:
            logger.warning("SAM2 inference failed: %s", exc)
            sam2_before_mask = np.zeros_like(algo_mask)

        # Reset and update overlays for this patch
        sam2_before[patch_idx] = 0.0
        algo_overlay[patch_idx] = 0.0
        sam2_after[patch_idx] = 0.0

        sam2_before[patch_idx, ay1:ay1 + crop_h, ax1:ax1 + crop_w] = sam2_before_mask
        algo_overlay[patch_idx, ay1:ay1 + crop_h, ax1:ax1 + crop_w] = algo_mask

        before_layer.data = sam2_before.copy()
        algo_layer.data = algo_overlay.copy()
        after_layer.data = sam2_after.copy()

        before_layer.visible = True
        algo_layer.visible = True
        after_layer.visible = False

        viewer.dims.set_current_step(0, patch_idx)

        pending["crop"] = crop
        pending["box_xyxy_in_crop"] = box_in_crop
        pending["algo_mask"] = algo_mask
        pending["crop_origin"] = (patch_idx, ay1, ax1, crop_h, crop_w)

        file_idx = patch_meta[patch_idx][0]
        file_name = Path(files[file_idx]).name
        fmeta = patch_meta[patch_idx]
        abs_y1 = fmeta[1] + y1
        abs_x1 = fmeta[2] + x1
        abs_y2 = fmeta[1] + y2
        abs_x2 = fmeta[2] + x2
        logger.info(
            "[ANNOTATION] Ready for accept: file=%s patch=%d box_in_patch=[y1=%d x1=%d y2=%d x2=%d] "
            "abs_coords=[y1=%d x1=%d y2=%d x2=%d]  algo_pixels=%d",
            file_name, patch_idx, y1, x1, y2, x2, abs_y1, abs_x1, abs_y2, abs_x2,
            int(algo_mask.sum()),
        )
        viewer.status = (
            f"Patch {patch_idx}  [{file_name}]  "
            f"box=[{y1},{x1}→{y2},{x2}]  algo pixels={int(algo_mask.sum())}  —  "
            "Blue=SAM2 current  Green=target  |  A = save & train   D = discard"
        )

    def _on_shape_data_changed(event=None) -> None:  # noqa: ARG001
        if not _auto_process_enabled[0]:
            return
        current_count = len(shapes_layer.data)
        if current_count > _prev_shape_count[0]:
            _prev_shape_count[0] = current_count
            on_process()
        elif current_count < _prev_shape_count[0]:
            # Shape was deleted — clear overlays for the pending annotation
            _clear_pending()
        else:
            _prev_shape_count[0] = current_count

    shapes_layer.events.data.connect(_on_shape_data_changed)

    # ---- Key binding: A = Accept, save, train, show update -----------------
    @viewer.bind_key("a")
    @viewer.bind_key("A")
    def on_accept(viewer: napari.Viewer) -> None:
        logger.info("[TRAIN] Accept key pressed.")
        crop = pending.get("crop")
        algo_mask = pending.get("algo_mask")
        box_in_crop = pending.get("box_xyxy_in_crop")
        origin = pending.get("crop_origin")
        if crop is None or algo_mask is None:
            logger.warning("[TRAIN] Accept pressed but no pending annotation was found.")
            viewer.status = "Nothing pending. Draw a rectangle first."
            return

        patch_idx, ay1, ax1, crop_h, crop_w = origin

        # 1. Save sample
        idx = sample_counter[0]
        fname = f"{idx}.tif"
        tifffile.imwrite(str(input_dir / fname), crop.astype(np.float32))
        tifffile.imwrite(str(target_dir / fname), algo_mask.astype(np.float32))
        sample_counter[0] += 1
        logger.info(
            "[TRAIN] Saved sample %d  crop=%s  mask_sum=%d",
            idx,
            crop.shape,
            int(algo_mask.sum()),
        )

        # 2. Train on all accumulated samples
        n_samples = len(list(input_dir.glob("*.tif")))
        viewer.status = (
            f"Saved sample {idx}. Training on {n_samples} sample(s) "
            f"for {train_epochs} epoch(s) — napari will be briefly unresponsive…"
        )
        try:
            logger.info(
                "[TRAIN] Launching training on %d sample(s) for %d epoch(s).",
                n_samples,
                train_epochs,
            )
            final_loss = _train_round(
                sam_model, refinement_head,
                input_dir, target_dir,
                lr=lr, batch_size=batch_size, epochs=train_epochs,
                device=device,
            )
        except Exception as exc:
            logger.exception("Training failed: %s", exc)
            viewer.status = f"Training error: {exc}"
            return

        # 3. Save updated weights
        torch.save(
            {
                "model_id": model_id,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "refinement_features": refinement_features,
                "trainable_state_dict": {
                    "sam_model": {
                        k: v for k, v in sam_model.state_dict().items()
                        if any(
                            req_p.data_ptr() == v.data_ptr()
                            for req_p in sam_model.parameters()
                            if req_p.requires_grad
                        )
                    },
                    "refinement_head": refinement_head.state_dict(),
                },
            },
            str(weights_path),
        )
        logger.info("[TRAIN] Saved updated adapter weights: %s", weights_path)

        # 4. Show updated SAM2 prediction
        try:
            sam2_after_mask = _predict_sam2(
                _predictor, refinement_head, crop, box_in_crop, device
            )
        except Exception as exc:
            logger.warning("Post-training inference failed: %s", exc)
            sam2_after_mask = np.zeros_like(algo_mask)

        sam2_after[patch_idx] = 0.0
        sam2_after[patch_idx, ay1:ay1 + crop_h, ax1:ax1 + crop_w] = sam2_after_mask
        after_layer.data = sam2_after.copy()
        after_layer.visible = True

        viewer.status = (
            f"Sample {idx} saved. Trained {train_epochs} epoch(s) on {n_samples} sample(s). "
            f"loss={final_loss:.4f}  |  "
            "Blue=SAM2 before  Green=target  Cyan=SAM2 after  |  Draw next box."
        )
        _clear_pending()

    # ---- Key binding: D = Discard ------------------------------------------
    @viewer.bind_key("d")
    @viewer.bind_key("D")
    def on_discard(viewer: napari.Viewer) -> None:
        _clear_pending()
        viewer.status = "Discarded. Draw a new box."

    # ---- Clear helper -------------------------------------------------------
    def _clear_pending() -> None:
        pidx = (pending.get("crop_origin") or (None,))[0]
        if pidx is not None:
            sam2_before[pidx] = 0.0
            algo_overlay[pidx] = 0.0
            sam2_after[pidx] = 0.0
            before_layer.data = sam2_before.copy()
            algo_layer.data = algo_overlay.copy()
            after_layer.data = sam2_after.copy()
        pending["crop"] = None
        pending["box_xyxy_in_crop"] = None
        pending["algo_mask"] = None
        pending["crop_origin"] = None
        before_layer.visible = False
        algo_layer.visible = False
        after_layer.visible = False
        _auto_process_enabled[0] = False
        shapes_layer.selected_data = set()
        shapes_layer.data = []
        _prev_shape_count[0] = 0
        _auto_process_enabled[0] = True

    napari.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive napari loop: draw a box → see SAM2 prediction + "
            "algorithmic target → accept to save and fine-tune → see updated "
            "prediction.  All in one session."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow per annotation
-----------------------
  1. Navigate to a patch (N-axis slider).
  2. Draw a rectangle in the 'Boxes' layer.
    3. Processing runs automatically and shows Blue (SAM2 current) and Green (algorithmic target).
  4. A  — saves the sample, trains for --train-epochs epochs on all
          accumulated samples, then shows Cyan (SAM2 updated).
  5. D  — discards and clears overlays.

Adapter weights are saved after every accepted sample so you can quit
and resume without losing progress.

Example YAML config for run_pipeline.exe:
---
run:
- name: Interactive SAM2 training session
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_interactive_train.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'
  - --model-id: facebook/sam2-hiera-large

- name: Interactive SAM2 training (custom settings)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_interactive_train.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'
  - --model-id: facebook/sam2-hiera-large
  - --patch-size: 512
  - --channel: 0
  - --margin: 30
  - --tolerance: 1.0
  - --lora-rank: 16
  - --train-epochs: 15
  - --lr: 1e-4
  - --batch-size: 4

- name: Pause before full offline training
  type: pause
  message: 'Review collected samples, then continue for full training.'

- name: Full offline training on collected samples
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_macropinosome_training.py'
  - --input-folder: '%YAML%/deep_learning_output'
  - --max-epochs: 100

- name: Stop intentionally
  type: stop
  message: 'Pipeline stopped intentionally.'
""",
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Glob pattern for input timelapse files (e.g. '**/*.ome.tif').",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help=(
            "Root folder for training data and adapter weights.  "
            "training_set/ and sam2_adapter_weights.pt are written here."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam2-hiera-large",
        help="HuggingFace SAM2 model ID (default: facebook/sam2-hiera-large).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Square patch size for grid extraction (default: 512).",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Limit total patches shown in napari (default: all).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to use (0-based, default: 0).",
    )
    parser.add_argument(
        "--z-index",
        type=int,
        default=-1,
        help="Z-slice index; -1 = middle slice (default).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Extra pixels added around the drawn box for crop extraction (default: 20).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="±tolerance for the algorithmic threshold step (default: 0.0).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA adapter rank (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=1.0,
        help="LoRA scaling alpha (default: 1.0).",
    )
    parser.add_argument(
        "--refinement-features",
        type=int,
        default=32,
        help="Feature channels in the per-frame refinement CNN head (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA and decoder params (default: 1e-4).  "
             "Refinement head uses 5× this value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the quick training round (default: 4).",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=10,
        help=(
            "Epochs to run on each accept.  "
            "As the dataset grows, more epochs are useful.  Default: 10."
        ),
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

    for noisy_logger in (
        "napari",
        "vispy",
        "qtpy",
        "matplotlib",
        "httpx",
        "httpcore",
        "huggingface_hub",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger.info(
        "Training visibility: logs are tagged with [TRAIN]. Draw rectangle, then press A in napari canvas."
    )

    search_subfolders = "**" in args.input_search_pattern
    files = sorted(
        rp.get_files_to_process2(
            args.input_search_pattern,
            search_subfolders=search_subfolders,
        )
    )
    if not files:
        logger.error("No files matched pattern: %s", args.input_search_pattern)
        raise SystemExit(1)

    logger.info("Found %d file(s):", len(files))
    for f in files:
        logger.info("  %s", os.path.basename(f))

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    run_interactive_train(
        files=files,
        output_folder=output_folder,
        channel=args.channel,
        z_index=args.z_index,
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        margin=args.margin,
        tolerance=args.tolerance,
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        refinement_features=args.refinement_features,
        lr=args.lr,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
    )


if __name__ == "__main__":
    main()
