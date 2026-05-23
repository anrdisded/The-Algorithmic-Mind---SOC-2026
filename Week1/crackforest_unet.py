"""
CrackForest Segmentation — Attention Residual U-Net with 3-Fold Cross-Validation
===================================================================================
Architecture:
  - Residual encoder blocks (ResNet-style skip connections within each block)
  - Attention gates on skip connections (Schlemper et al., 2019)
  - Deep supervision heads at multiple decoder scales
  - ASPP bottleneck (Atrous Spatial Pyramid Pooling)
  - Squeeze-and-Excitation channel attention in each block

Loss:
  - Combined Dice + Focal Binary Cross-Entropy (handles extreme class imbalance)
  - Deep supervision auxiliary losses weighted at 0.4 and 0.2

Augmentation (albumentations):
  - Geometric: flip, rotate, elastic/grid distortion, perspective
  - Color: brightness/contrast, CLAHE, hue-saturation, coarse dropout
  - Noise: Gaussian noise, blur

Usage:
  python crackforest_unet.py --data_dir /path/to/crackforest_extracted --epochs 100

  The zip should extract to:
    crackforest/
      image/   <- RGB images (or grayscale)
      mask/    <- Binary masks (0/255 or 0/1)
"""

import os
import argparse
import random
import warnings
import math
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class Config:
    IMG_SIZE   = 448          # resize (H, W) — must be divisible by 32
    IN_CH      = 3
    NUM_CLS    = 1            # binary segmentation
    BASE_CH    = 32           # encoder base channels
    BATCH      = 8
    EPOCHS     = 100
    LR         = 3e-4
    WEIGHT_DECAY = 1e-4
    K_FOLDS    = 3
    AMP        = True         # mixed precision
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    THRESH     = 0.5
    PATIENCE   = 15           # early stopping patience
    SCHEDULER  = "cosine"     # "cosine" | "plateau"
    SAVE_DIR   = "./checkpoints"

cfg = Config()

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
TRAIN_AUG = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6,
                       border_mode=cv2.BORDER_REFLECT),
    A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.3),
    A.GridDistortion(p=0.3),
    A.Perspective(p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.CLAHE(clip_limit=4.0, p=1.0),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], p=0.7),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                    min_holes=1, fill_value=0, mask_fill_value=0, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

VAL_AUG = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class CrackDataset(Dataset):
    MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, image_dir, mask_dir, indices=None, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform

        all_imgs = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.MASK_EXTS
        ])
        self.images = [all_imgs[i] for i in indices] if indices is not None else all_imgs

    def __len__(self):
        return len(self.images)

    def _find_mask(self, img_path):
        """Try several naming conventions to locate the paired mask."""
        stem = img_path.stem
        for ext in [".png", ".bmp", ".jpg", ".jpeg"]:
            for suffix in ["", "_mask", "_gt", "_label"]:
                candidate = self.mask_dir / f"{stem}{suffix}{ext}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"No mask found for {img_path.name} in {self.mask_dir}")

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self._find_mask(img_path)

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)   # binarize

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].unsqueeze(0).float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.
            mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


# ─────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Two conv layers with residual shortcut + SE attention."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se      = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.skip    = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        return self.relu(out + identity)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling bottleneck."""
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for r in rates
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        size = x.shape[-2:]
        feats = [self.conv1x1(x)]
        feats += [a(x) for a in self.atrous]
        gap   = F.interpolate(self.gap(x), size=size, mode="bilinear", align_corners=False)
        feats.append(gap)
        return self.project(torch.cat(feats, dim=1))


class AttentionGate(nn.Module):
    """Additive attention gate (Schlemper et al., 2019)."""
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.Wx = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from lower level), x: skip connection
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        # match spatial dims
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


# ─────────────────────────────────────────────
# Attention Residual U-Net
# ─────────────────────────────────────────────

class AttResUNet(nn.Module):
    def __init__(self, in_ch=3, num_cls=1, base_ch=32):
        super().__init__()
        c = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]  # [32,64,128,256,512]

        # ── Encoder ──
        self.enc1 = ResidualBlock(in_ch,  c[0])
        self.enc2 = ResidualBlock(c[0],   c[1])
        self.enc3 = ResidualBlock(c[1],   c[2])
        self.enc4 = ResidualBlock(c[2],   c[3])
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck (ASPP) ──
        self.bottleneck = ASPP(c[3], c[4])

        # ── Attention Gates ──
        self.att4 = AttentionGate(c[4], c[3], c[3] // 2)
        self.att3 = AttentionGate(c[3], c[2], c[2] // 2)
        self.att2 = AttentionGate(c[2], c[1], c[1] // 2)
        self.att1 = AttentionGate(c[1], c[0], c[0] // 2)

        # ── Decoder ──
        self.up4   = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4  = ResidualBlock(c[3]*2, c[3])

        self.up3   = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3  = ResidualBlock(c[2]*2, c[2])

        self.up2   = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2  = ResidualBlock(c[1]*2, c[1])

        self.up1   = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1  = ResidualBlock(c[0]*2, c[0])

        # ── Segmentation heads ──
        self.head_main = nn.Conv2d(c[0], num_cls, 1)
        self.head_aux1 = nn.Conv2d(c[1], num_cls, 1)  # deep supervision at 1/2 scale
        self.head_aux2 = nn.Conv2d(c[2], num_cls, 1)  # deep supervision at 1/4 scale

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                   # /1
        e2 = self.enc2(self.pool(e1))       # /2
        e3 = self.enc3(self.pool(e2))       # /4
        e4 = self.enc4(self.pool(e3))       # /8

        # Bottleneck
        b  = self.bottleneck(self.pool(e4)) # /16

        # Decoder with attention gates
        d4 = self.dec4(torch.cat([self.up4(b), self.att4(b, e4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(d4, e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(d3, e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(d2, e1)], dim=1))

        # Segmentation outputs
        out_main = self.head_main(d1)
        out_aux1 = self.head_aux1(d2)   # 1/2 scale
        out_aux2 = self.head_aux2(d3)   # 1/4 scale

        if self.training:
            return out_main, out_aux1, out_aux2
        return out_main


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    flat_p = pred.view(-1)
    flat_t = target.view(-1)
    inter  = (flat_p * flat_t).sum()
    return 1 - (2 * inter + smooth) / (flat_p.sum() + flat_t.sum() + smooth)


def focal_bce_loss(pred, target, gamma=2.0, alpha=0.25):
    bce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    prob = torch.sigmoid(pred)
    pt   = target * prob + (1 - target) * (1 - prob)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def combined_loss(pred, target, dice_w=0.5, focal_w=0.5):
    return dice_w * dice_loss(pred, target) + focal_w * focal_bce_loss(pred, target)


def deep_supervision_loss(outputs, target, weights=(1.0, 0.4, 0.2)):
    main, aux1, aux2 = outputs
    H, W = target.shape[-2:]

    # resize aux predictions to match full target size for loss
    aux1_up = F.interpolate(aux1, size=(H, W), mode="bilinear", align_corners=False)
    aux2_up = F.interpolate(aux2, size=(H, W), mode="bilinear", align_corners=False)

    loss  = weights[0] * combined_loss(main,     target)
    loss += weights[1] * combined_loss(aux1_up,  target)
    loss += weights[2] * combined_loss(aux2_up,  target)
    return loss


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(preds, targets, thresh=cfg.THRESH):
    """preds: logits tensor, targets: binary tensor"""
    preds_bin = (torch.sigmoid(preds) > thresh).long().cpu().numpy().flatten()
    targets_np = targets.long().cpu().numpy().flatten()

    tp = ((preds_bin == 1) & (targets_np == 1)).sum()
    fp = ((preds_bin == 1) & (targets_np == 0)).sum()
    fn = ((preds_bin == 0) & (targets_np == 1)).sum()
    tn = ((preds_bin == 0) & (targets_np == 0)).sum()

    eps  = 1e-7
    iou  = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)

    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────
# Training / Validation Loops
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_masks = [], []

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast(enabled=cfg.AMP):
            outputs = model(imgs)
            loss    = deep_supervision_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.append(outputs[0].detach())
        all_masks.append(masks.detach())

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_masks))
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast(enabled=cfg.AMP):
            logits = model(imgs)
            loss   = combined_loss(logits, masks)
        total_loss += loss.item()
        all_preds.append(logits.detach())
        all_masks.append(masks.detach())

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_masks))
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ─────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────

def save_prediction_grid(model, dataset, device, fold, n=6):
    model.eval()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask = dataset[idx]
            logit = model(img.unsqueeze(0).to(device))
            pred  = (torch.sigmoid(logit) > cfg.THRESH).squeeze().cpu().numpy()
            mask  = mask.squeeze().numpy()

            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * std + mean).clip(0, 1)

            axes[row, 0].imshow(img_np); axes[row, 0].set_title("Image")
            axes[row, 1].imshow(mask, cmap="gray"); axes[row, 1].set_title("GT Mask")
            axes[row, 2].imshow(pred, cmap="gray"); axes[row, 2].set_title("Prediction")
            for ax in axes[row]: ax.axis("off")

    plt.suptitle(f"Fold {fold} Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{cfg.SAVE_DIR}/fold{fold}_predictions.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction grid → {path}")


def plot_fold_history(history, fold):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_dice"], label="Train")
    axes[1].plot(epochs, history["val_dice"],   label="Val")
    axes[1].set_title("Dice"); axes[1].legend()

    axes[2].plot(epochs, history["train_iou"], label="Train")
    axes[2].plot(epochs, history["val_iou"],   label="Val")
    axes[2].set_title("IoU"); axes[2].legend()

    plt.suptitle(f"Fold {fold} Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{cfg.SAVE_DIR}/fold{fold}_history.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved history plot → {path}")


# ─────────────────────────────────────────────
# Main: 3-Fold Cross-Validation
# ─────────────────────────────────────────────

def run_cv(image_dir, mask_dir):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE)
    print(f"Device: {device}")

    # Collect all image indices
    full_dataset = CrackDataset(image_dir, mask_dir, transform=None)
    n = len(full_dataset)
    print(f"Total images: {n}")
    assert n >= cfg.K_FOLDS * 2, f"Too few images ({n}) for {cfg.K_FOLDS}-fold CV"

    kf = KFold(n_splits=cfg.K_FOLDS, shuffle=True, random_state=SEED)
    indices = np.arange(n)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{cfg.K_FOLDS}  |  train={len(train_idx)}, val={len(val_idx)}")
        print(f"{'='*60}")

        train_ds = CrackDataset(image_dir, mask_dir, indices=train_idx, transform=TRAIN_AUG)
        val_ds   = CrackDataset(image_dir, mask_dir, indices=val_idx,   transform=VAL_AUG)

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH, shuffle=False,
                                  num_workers=4, pin_memory=True)

        model     = AttResUNet(cfg.IN_CH, cfg.NUM_CLS, cfg.BASE_CH).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                      weight_decay=cfg.WEIGHT_DECAY)

        if cfg.SCHEDULER == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=7, verbose=True)

        scaler = GradScaler(enabled=cfg.AMP)

        best_dice    = 0.0
        patience_cnt = 0
        history = {k: [] for k in
                   ["train_loss","val_loss","train_dice","val_dice","train_iou","val_iou"]}

        for epoch in range(1, cfg.EPOCHS + 1):
            tr  = train_one_epoch(model, train_loader, optimizer, scaler, device)
            val = validate(model, val_loader, device)

            if cfg.SCHEDULER == "cosine":
                scheduler.step()
            else:
                scheduler.step(val["dice"])

            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(val["loss"])
            history["train_dice"].append(tr["dice"])
            history["val_dice"].append(val["dice"])
            history["train_iou"].append(tr["iou"])
            history["val_iou"].append(val["iou"])

            print(f"  Ep {epoch:03d}/{cfg.EPOCHS} "
                  f"| Tr Loss {tr['loss']:.4f} Dice {tr['dice']:.4f} IoU {tr['iou']:.4f} "
                  f"| Val Loss {val['loss']:.4f} Dice {val['dice']:.4f} IoU {val['iou']:.4f}")

            if val["dice"] > best_dice:
                best_dice    = val["dice"]
                patience_cnt = 0
                ckpt_path    = f"{cfg.SAVE_DIR}/fold{fold}_best.pth"
                torch.save({"epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_dice": best_dice,
                            "cfg": cfg.__dict__}, ckpt_path)
                print(f"    ✓ Best model saved (Dice={best_dice:.4f})")
            else:
                patience_cnt += 1
                if patience_cnt >= cfg.PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # ── Load best model and evaluate ──
        ckpt = torch.load(f"{cfg.SAVE_DIR}/fold{fold}_best.pth", map_location=device)
        model.load_state_dict(ckpt["model_state"])

        final_val = validate(model, val_loader, device)
        fold_results.append(final_val)
        print(f"\n  Fold {fold} Final → "
              f"Dice={final_val['dice']:.4f}  IoU={final_val['iou']:.4f}  "
              f"F1={final_val['f1']:.4f}  Prec={final_val['precision']:.4f}  "
              f"Rec={final_val['recall']:.4f}")

        save_prediction_grid(model, val_ds, device, fold)
        plot_fold_history(history, fold)

    # ── Cross-Validation Summary ──
    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    metrics_keys = ["dice", "iou", "f1", "precision", "recall"]
    cv_scores = {k: [r[k] for r in fold_results] for k in metrics_keys}
    for k in metrics_keys:
        vals = cv_scores[k]
        print(f"  {k.upper():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
              f"(folds: {[f'{v:.4f}' for v in vals]})")

    # Save summary
    summary_path = f"{cfg.SAVE_DIR}/cv_summary.txt"
    with open(summary_path, "w") as f:
        f.write("CrackForest — Attention Residual U-Net 3-Fold CV Summary\n")
        f.write("="*60 + "\n")
        for k in metrics_keys:
            vals = cv_scores[k]
            f.write(f"{k.upper():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n")
    print(f"\n  Summary saved → {summary_path}")

    return cv_scores


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attention Residual U-Net for CrackForest with 3-Fold CV")
    parser.add_argument("--data_dir", type=str, required=True,
        help="Root dir of the extracted CrackForest dataset (contains image/ and mask/)")
    parser.add_argument("--image_subdir", type=str, default="image")
    parser.add_argument("--mask_subdir",  type=str, default="mask")
    parser.add_argument("--epochs",   type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch",    type=int, default=cfg.BATCH)
    parser.add_argument("--base_ch",  type=int, default=cfg.BASE_CH)
    parser.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    parser.add_argument("--lr",       type=float, default=cfg.LR)
    parser.add_argument("--save_dir", type=str, default=cfg.SAVE_DIR)
    args = parser.parse_args()

    cfg.EPOCHS   = args.epochs
    cfg.BATCH    = args.batch
    cfg.BASE_CH  = args.base_ch
    cfg.IMG_SIZE = args.img_size
    cfg.LR       = args.lr
    cfg.SAVE_DIR = args.save_dir

    image_dir = os.path.join(args.data_dir, args.image_subdir)
    mask_dir  = os.path.join(args.data_dir, args.mask_subdir)

    assert os.path.isdir(image_dir), f"Image dir not found: {image_dir}"
    assert os.path.isdir(mask_dir),  f"Mask dir not found: {mask_dir}"

    run_cv(image_dir, mask_dir)
