import argparse
import os
import random
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset

import utils
from networks.hybrid_seg_modeling import QuantumUNet
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
from torchinfo import summary


# =========================
# Dataset
# =========================
class StreamingGeospatialDataset(IterableDataset):
    def __init__(self, imagery_fns, lr_label_fns=None,
                 hr_label_fns=None, chip_size=256, num_chips_per_tile=100,
                 image_transform=None, label_transform=None,
                 nodata_check=None):
        self.fns = list(zip(
            imagery_fns,
            lr_label_fns if lr_label_fns is not None else [None] * len(imagery_fns),
            hr_label_fns if hr_label_fns is not None else [None] * len(imagery_fns),
        ))
        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

    def __iter__(self):
        for img_fn, lr_label_fn, hr_label_fn in self.fns:
            with rasterio.open(img_fn, "r") as img_fp:
                lr_fp = rasterio.open(lr_label_fn, "r") if lr_label_fn is not None else None
                hr_fp = rasterio.open(hr_label_fn, "r") if hr_label_fn is not None else None

                height, width = img_fp.shape
                chips_yielded = 0

                while chips_yielded < self.num_chips_per_tile:
                    x = np.random.randint(0, width - self.chip_size)
                    y = np.random.randint(0, height - self.chip_size)

                    # read image
                    img = np.rollaxis(
                        img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3
                    )

                    # normalize
                    if self.image_transform is not None:
                        img_norm = self.image_transform(img)
                    else:
                        img_norm = torch.from_numpy(
                            np.moveaxis(img.astype(np.float32), -1, 0)
                        )

                    # raw image (for recon loss)
                    img_raw = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

                    # LR labels
                    lr_labels = None
                    if lr_fp is not None:
                        lr_labels = lr_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                        if self.label_transform is not None:
                            lr_labels = self.label_transform(lr_labels)

                    # HR labels
                    hr_labels = None
                    if hr_fp is not None:
                        hr_labels = hr_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                        if self.label_transform is not None:
                            hr_labels = self.label_transform(hr_labels)

#                    if hr_labels is not None:
#                        yield img_norm, lr_labels, img_raw, hr_labels
#                    elif lr_labels is not None:
#                        yield img_norm, lr_labels, img_raw
#                    else:
#                        yield img_norm, img_raw  # pure recon
                    label_to_use = hr_labels if hr_labels is not None else lr_labels
                    if label_to_use is not None:
                        yield img_norm, label_to_use, img_raw
                    else:
                        yield img_norm, img_raw  # pure recon


                    chips_yielded += 1


# =========================
# Losses
# =========================
#def weighted_dice_loss(inputs, targets, ignore_index=0, eps=1e-6):
#    num_classes = inputs.shape[1]
#    probs = F.softmax(inputs, dim=1)
#
#    mask = (targets != ignore_index).unsqueeze(1).float()
#    probs = probs * mask
#    onehot = F.one_hot(targets.clamp(min=0),
#                       num_classes=num_classes).permute(0, 3, 1, 2).float()
#    onehot = onehot * mask
#
#    dims = (0, 2, 3)
#    intersection = torch.sum(probs * onehot, dims)
#    cardinality = torch.sum(probs + onehot, dims)
#    dice = (2.0 * intersection + eps) / (cardinality + eps)
#    return (1.0 - dice).mean()
#
#
#class HybridSegLoss(nn.Module):
#    def __init__(self, ce_weight=0.3, dice_weight=0.7, ignore_index=0):
#        super().__init__()
#        self.ce_weight = ce_weight
#        self.dice_weight = dice_weight
#        self.ignore_index = ignore_index
#
#    def forward(self, inputs, targets):
#        ce = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)
#        dice = weighted_dice_loss(inputs, targets, ignore_index=self.ignore_index)
#        return self.ce_weight * ce + self.dice_weight * dice
def weighted_dice_loss(inputs, targets, ignore_index=0, eps=1e-6):
    """
    NaN-safe Dice:
    - ignores pixels with targets==ignore_index
    - ignores classes that have zero valid pixels in the batch
    """
    # inputs: [B,C,H,W], targets: [B,H,W] (int64)
    C = inputs.shape[1]
    probs = F.softmax(inputs, dim=1)

    valid = (targets != ignore_index)  # [B,H,W]
    if valid.sum() == 0:
        # no valid pixels at all -> return 0 (don't break training)
        return inputs.new_tensor(0.0)

    # one-hot only over valid pixels
    onehot = F.one_hot(targets.clamp(min=0), num_classes=C)  # [B,H,W,C]
    onehot = onehot.permute(0, 3, 1, 2).float()              # [B,C,H,W]

    probs = probs * valid.unsqueeze(1)    # mask probs
    onehot = onehot * valid.unsqueeze(1)  # mask onehot

    dims = (0, 2, 3)
    intersection = torch.sum(probs * onehot, dims)          # [C]
    cardinality  = torch.sum(probs + onehot, dims)          # [C]

    # Only average over classes that actually appear or have probs
    present = cardinality > 0
    if present.sum() == 0:
        return inputs.new_tensor(0.0)

    dice = (2.0 * intersection[present] + eps) / (cardinality[present] + eps)
    loss = (1.0 - dice).mean()
    if torch.isnan(loss):
        return inputs.new_tensor(0.0)
    return loss


class HybridSegLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.7, ignore_index=0, ce_label_smoothing=0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.ce_label_smoothing = ce_label_smoothing

        # Cross-entropy initialization
        try:
            self.ce_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_index, 
                label_smoothing=ce_label_smoothing
            )
        except TypeError:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # Safety checks: replace NaNs/Infs
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e4, neginf=-1e4)
        targets = torch.clamp(targets, 0, inputs.shape[1]-1).long()
        ce_val = self.ce_loss(inputs, targets)
        dice_val = weighted_dice_loss(inputs, targets, ignore_index=self.ignore_index)
        loss = self.ce_weight * ce_val + self.dice_weight * dice_val
        if torch.isnan(loss) or torch.isinf(loss):
            return inputs.new_tensor(0.0)

        return loss


# === SSIM helper (channel-wise, works on 4-band tensors in [0,1]) ===
import torch.nn.functional as _F

def _gaussian_window(ws, sigma, C, device):
    x = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g1 = torch.exp(-(x**2) / (2 * sigma**2))
    g1 = (g1 / g1.sum()).unsqueeze(0)
    w2 = (g1.t() @ g1).unsqueeze(0).unsqueeze(0)           # 1x1xHxW
    return w2.repeat(C, 1, 1, 1)                           # Cx1xHxW

def ssim_torch(x, y, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    # x,y: BxCxHxW in [0,1]
    C = x.size(1)
    w = _gaussian_window(window_size, sigma, C, x.device)
    mu_x = _F.conv2d(x, w, padding=window_size//2, groups=C)
    mu_y = _F.conv2d(y, w, padding=window_size//2, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sigma_x2 = _F.conv2d(x*x, w, padding=window_size//2, groups=C) - mu_x2
    sigma_y2 = _F.conv2d(y*y, w, padding=window_size//2, groups=C) - mu_y2
    sigma_xy = _F.conv2d(x*y, w, padding=window_size//2, groups=C) - mu_xy
    num  = (2*mu_xy + C1) * (2*sigma_xy + C2)
    den  = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / (den + 1e-12)).mean()

# =========================
# Phase 1: Train reconstruction + aux
# =========================
def save_recon_grid(imgs_raw, recon, epoch, save_dir, nrow=4):
    """
    Save a grid of raw vs recon for first few samples
    """
    raw = imgs_raw.detach().cpu().numpy()
    rec = recon.detach().cpu().numpy()
    B = min(nrow, raw.shape[0])

    fig, axs = plt.subplots(2, B, figsize=(3*B, 6))
    for i in range(B):
        axs[0, i].imshow(np.clip(np.transpose(raw[i], (1, 2, 0))[..., :3], 0, 1))
        axs[0, i].set_title("Raw")
        axs[0, i].axis("off")
        axs[1, i].imshow(np.clip(np.transpose(rec[i], (1, 2, 0))[..., :3], 0, 1))
        axs[1, i].set_title("Recon")
        axs[1, i].axis("off")

    os.makedirs(save_dir, exist_ok=True)
    out_fn = os.path.join(save_dir, f"recon_grid_epoch_{epoch}.png")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=150)
    plt.close(fig)
    print(f"Saved recon grid {out_fn}")

# =========================
# Phase 1: Image Reconstruction + Auxilliary Training
# =========================
def train_recon(args, model, device):
    df = pd.read_csv(args.list_dir)

    dataset = StreamingGeospatialDataset(
        imagery_fns=df["image_fn"].values,
        lr_label_fns=df["label_fn"].values if "label_fn" in df.columns else None,
        chip_size=args.chip_size,
        num_chips_per_tile=args.num_chips_per_tile,
        image_transform=lambda x: torch.from_numpy(
            ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
        ).permute(2, 0, 1),
        label_transform=lambda y: torch.from_numpy(
            np.take(utils.LABEL_CLASS_TO_IDX_MAP, y, mode="clip").astype(np.int64)
        ),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=True, prefetch_factor=4)

    l1 = nn.L1Loss()
    # label smoothing helps calibration of aux logits (better pseudo labels)
    try:
        aux_ce = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.aux_label_smoothing)
    except TypeError:
        aux_ce = nn.CrossEntropyLoss(ignore_index=0)

    # start by training everything
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

    model.train()
    # ==== Initial Layer Status ====
    print("\n==== Initial Layer Status ====")
    for name, param in model.named_parameters():
        print(f"{name:60s} | requires_grad={param.requires_grad}")
    print("==============================\n")
    # ========

    for epoch in range(args.epochs_recon):
        # Freeze recon branch at the chosen epoch and refocus on aux head
        if epoch == args.freeze_recon_epoch:
            print(">>> Freezing reconstruction head and refocusing on auxiliary classifier.")
            for p in model.recon_conv.parameters():
                p.requires_grad = False
            # Rebuild optimizer to exclude frozen params
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.base_lr)
                                   
            # ==== Verify which layers remain trainable after freezing ====
            print(f"\n==== Layer Status After Freezing (Epoch {epoch}) ====")
            trainable, frozen = [], []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable.append(name)
                else:
                    frozen.append(name)
        
            print(f"Trainable layers ({len(trainable)}):")
            for n in trainable:
                print("   ", n)
        
            print(f"Frozen layers ({len(frozen)}):")
            for n in frozen:
                print("   ", n)
            print("===========================================\n")
            # ==== Verify which layers remain trainable after freezing ====           
                                   
                                   
        # Dynamic loss weighting: after freeze, drop recon loss and boost aux loss
        recon_w = args.recon_weight if epoch < args.freeze_recon_epoch else 0.0
        aux_w   = args.aux_weight if epoch < args.freeze_recon_epoch else args.aux_weight_after_freeze

        recon_losses, aux_losses, total_losses = [], [], []
        pbar = tqdm(loader, desc=f"Recon Epoch {epoch}", leave=True,
                    total=args.num_chips_per_tile * len(df) // args.batch_size)

        for step, batch in enumerate(pbar):
            if len(batch) == 3:
                imgs_norm, lr_labels, imgs_raw = batch
            else:
                imgs_norm, lr_labels, imgs_raw, _ = batch

            imgs_norm = imgs_norm.to(device)
            imgs_raw  = imgs_raw.to(device).float() / 255.0
            lr_labels = lr_labels.to(device)

            seg_logits, recon, aux_logits = model(imgs_norm)

            recon_pred = torch.sigmoid(recon)
            # SSIM-augmented reconstruction loss (keeps edges/detail)
            loss_l1   = l1(recon_pred, imgs_raw)
            loss_ssim = 1.0 - ssim_torch(recon_pred, imgs_raw)
            loss_recon = 0.8 * loss_l1 + 0.2 * loss_ssim

            loss_aux = aux_ce(aux_logits, lr_labels)

            loss = recon_w * loss_recon + aux_w * loss_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            recon_losses.append(loss_recon.item())
            aux_losses.append(loss_aux.item())
            total_losses.append(loss.item())

            pbar.set_postfix({
                "recon": f"{np.mean(recon_losses):.4f}",
                "auxCE": f"{np.mean(aux_losses):.4f}",
                "total": f"{np.mean(total_losses):.4f}",
                "phase": "pre-freeze" if epoch < args.freeze_recon_epoch else "aux-focus"
            })

        # save checkpoint per epoch
        ckpt = os.path.join(args.savepath, f"recon_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint {ckpt}")

        # one figure per epoch (grid)
        with torch.no_grad():
            sample_recon = torch.sigmoid(recon)
            save_recon_grid(imgs_raw, sample_recon, epoch,
                            os.path.join(args.savepath, "recon_samples"))
#def train_recon(args, model, device):
#    df = pd.read_csv(args.list_dir)
#
#    dataset = StreamingGeospatialDataset(
#        imagery_fns=df["image_fn"].values,
#        lr_label_fns=df["label_fn"].values if "label_fn" in df.columns else None,
#        chip_size=args.chip_size,
#        num_chips_per_tile=args.num_chips_per_tile,
#        image_transform=lambda x: torch.from_numpy(
#            ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
#        ).permute(2, 0, 1),
#        label_transform=lambda y: torch.from_numpy(
#            np.take(utils.LABEL_CLASS_TO_IDX_MAP, y, mode="clip").astype(np.int64)
#        ),
#    )
#
#    loader = DataLoader(
#        dataset,
#        batch_size=args.batch_size,
#        num_workers=args.num_workers,
#        drop_last=True,
#        pin_memory=True,
#        prefetch_factor=4
#    )
#
#    l1 = nn.L1Loss()
#    try:
#        aux_ce = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.aux_label_smoothing)
#    except TypeError:
#        aux_ce = nn.CrossEntropyLoss(ignore_index=0)
#
#    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
#    model.train()
#
#    for epoch in range(args.epochs_recon):
#        recon_losses, aux_losses, total_losses = [], [], []
#        pbar = tqdm(loader, desc=f"Recon Epoch {epoch}", leave=True,
#                    total=args.num_chips_per_tile * len(df) // args.batch_size)
#
#        for step, batch in enumerate(pbar):
#            if len(batch) == 3:
#                imgs_norm, lr_labels, imgs_raw = batch
#            else:
#                imgs_norm, lr_labels, imgs_raw, _ = batch
#
#            imgs_norm = imgs_norm.to(device)
#            imgs_raw = imgs_raw.to(device).float() / 255.0
#            lr_labels = lr_labels.to(device)
#
#            seg_logits, recon, aux_logits = model(imgs_norm)
#
#            recon_pred = torch.sigmoid(recon)
#            loss_l1 = l1(recon_pred, imgs_raw)
#            loss_ssim = 1.0 - ssim_torch(recon_pred, imgs_raw)
#            loss_recon = 0.8 * loss_l1 + 0.2 * loss_ssim
#
#            loss_aux = aux_ce(aux_logits, lr_labels)
#
#            # Train both heads simultaneously, no freezing
#            loss = args.recon_weight * loss_recon + args.aux_weight * loss_aux
#
#            optimizer.zero_grad(set_to_none=True)
#            loss.backward()
#            if args.max_grad_norm > 0:
#                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#            optimizer.step()
#
#            recon_losses.append(loss_recon.item())
#            aux_losses.append(loss_aux.item())
#            total_losses.append(loss.item())
#
#            pbar.set_postfix({
#                "recon": f"{np.mean(recon_losses):.4f}",
#                "auxCE": f"{np.mean(aux_losses):.4f}",
#                "total": f"{np.mean(total_losses):.4f}",
#            })
#
#        ckpt = os.path.join(args.savepath, f"recon_epoch_{epoch}.pth")
#        torch.save(model.state_dict(), ckpt)
#        print(f"Saved checkpoint {ckpt}")
#
#        with torch.no_grad():
#            sample_recon = torch.sigmoid(recon)
#            save_recon_grid(imgs_raw, sample_recon, epoch,
#                            os.path.join(args.savepath, "recon_samples"))


# =========================
# Phase 2: Generate pseudo labels
# =========================
def generate_pseudo_labels(args, model, device):
    df = pd.read_csv(args.list_dir)
    out_dir = os.path.join(args.savepath, "pseudo_labels")
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating pseudo", leave=True):
            img_fn, lr_label_fn = row["image_fn"], row["label_fn"]

            with rasterio.open(img_fn) as img_fp:
                img = np.moveaxis(img_fp.read(), 0, 2)
                H, W = img_fp.height, img_fp.width
                profile_hr = img_fp.profile
                transform_hr, crs_hr = img_fp.transform, img_fp.crs

            with rasterio.open(lr_label_fn) as lr_fp:
                lr = lr_fp.read(1)
                transform_lr, crs_lr = lr_fp.transform, lr_fp.crs

            # Normalize and run through model
            img_norm = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
            img_norm = torch.from_numpy(
                np.moveaxis(img_norm, -1, 0).astype(np.float32)
            ).unsqueeze(0).to(device)

            # Temperature-scaled logits sharpen probabilities (better separation)
            _, _, aux_logits = model(img_norm)
            probs = F.softmax(aux_logits / args.aux_temp, dim=1)  # aux_temp < 1.0 sharpens
            conf, pseudo = torch.max(probs, dim=1)
            conf, pseudo = conf.squeeze(0), pseudo.squeeze(0)

            # Reproject LR labels to HR grid
            lr_up = np.zeros((H, W), dtype=np.uint8)
            reproject(
                source=lr,
                destination=lr_up,
                src_transform=transform_lr,
                src_crs=crs_lr,
                dst_transform=transform_hr,
                dst_crs=crs_hr,
                resampling=Resampling.nearest,
            )
            lr_up = np.take(utils.LABEL_CLASS_TO_IDX_MAP, lr_up, mode="clip")
            #lr_valid = torch.from_numpy(((lr_up != 0) & (lr_up != 4)).astype(np.uint8)).to(device)
            lr_valid = torch.from_numpy((lr_up != 0).astype(np.uint8)).to(device)


            # Align shapes
            min_h, min_w = min(conf.shape[0], lr_valid.shape[0]), min(conf.shape[1], lr_valid.shape[1])
            conf, pseudo, lr_valid = conf[:min_h, :min_w], pseudo[:min_h, :min_w], lr_valid[:min_h, :min_w]

            # Keep strategy: (A) top-K by confidence OR (B) absolute threshold
            valid_mask = (lr_valid > 0)
            if args.pseudo_keep > 0:
                v = conf[valid_mask].float()
                if v.numel() == 0:
                    thr = torch.tensor(1.0, device=conf.device)
                else:
                    # Subsample to prevent OOM or "tensor too large" errors
                    max_sample = 2000000  # 2 million pixels is plenty for quantile estimation
                    if v.numel() > max_sample:
                        perm = torch.randperm(v.numel(), device=v.device)[:max_sample]
                        v = v[perm]
                    thr = torch.quantile(v, max(0.0, 1.0 - args.pseudo_keep))
                mask = (conf >= thr) & valid_mask

            else:
                mask = (conf > args.pseudo_thresh) & valid_mask

            masked_pseudo = torch.where(mask, pseudo, torch.zeros_like(pseudo))

            # To numpy
            pseudo_np = masked_pseudo.detach().cpu().numpy()

            # Optional light morphology to denoise (best-effort, safe if SciPy missing)
            if args.morph_open > 0:
                try:
                    from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
                    st = generate_binary_structure(2, 1)
                    for cls in range(1, args.num_classes):
                        m = (pseudo_np == cls).astype(np.uint8)
                        if m.sum() == 0: 
                            continue
                        m = binary_opening(m, structure=st, iterations=args.morph_open)
                        m = binary_closing(m, structure=st, iterations=max(1, args.morph_open // 2))
                        pseudo_np[(pseudo_np == cls) & (m == 0)] = 0
                except Exception as e:
                    print(f"  Morphology skipped: {e}")

            # Stats
            unique, counts = np.unique(pseudo_np, return_counts=True)
            class_hist = {int(u): int(c) for u, c in zip(unique, counts)}
            kept = int((pseudo_np > 0).sum())
            print(f"{os.path.basename(img_fn)}: kept {kept}/{H*W} ({100*kept/(H*W):.2f}%), hist={class_hist}")

            # Save GeoTIFF
            out_fn = os.path.join(out_dir, os.path.basename(img_fn).replace(".tif", "_pseudo.tif"))
            prof = profile_hr.copy()
            prof.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
            with rasterio.open(out_fn, "w", **prof) as dst:
                dst.write(pseudo_np.astype(np.uint8), 1)
                dst.write_colormap(1, utils.LABEL_IDX_COLORMAP)



# =========================
# Phase 3: Train segmentation head (with validation)
# =========================
from sklearn.metrics import jaccard_score, accuracy_score
def train_seg(args, model, device):
    # --- Training data (pseudo-labeled) ---
    pseudo_dir = os.path.join(args.savepath, "pseudo_labels")
    df = pd.read_csv(args.list_dir)
    df["label_fn"] = df["image_fn"].apply(
        lambda fn: os.path.join(pseudo_dir, os.path.basename(fn).replace(".tif", "_pseudo.tif"))
    )


    train_dataset = StreamingGeospatialDataset(
        imagery_fns=df["image_fn"].values,
        lr_label_fns=df["label_fn"].values,
        chip_size=args.chip_size,
        num_chips_per_tile=args.num_chips_per_tile,
        image_transform=lambda x: torch.from_numpy(
            ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
        ).permute(2, 0, 1),
        label_transform=lambda y: torch.from_numpy(
            np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, y, mode="clip").astype(np.int64)
            #np.take(utils.LABEL_CLASS_TO_IDX_MAP, y, mode="clip").astype(np.int64)
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
    )

    # --- Validation data (HR labels) ---
    val_csv = getattr(args, "val_list", "./dataset/CSV_list/C_NYC-val.csv")
    if not os.path.exists(val_csv):
        print(f"[Warning] Validation CSV not found: {val_csv}. Skipping validation.")
        val_loader = None
    else:
        df_val = pd.read_csv(val_csv)
        required_cols = {"image_fn", "hr_label_fn"}
        missing = required_cols - set(df_val.columns)
        if missing:
            raise KeyError(f"Validation CSV missing columns: {missing}. Found: {list(df_val.columns)}")
    
        val_dataset = StreamingGeospatialDataset(
            imagery_fns=df_val["image_fn"].values,
            hr_label_fns=df_val["hr_label_fn"].values,
            chip_size=args.chip_size,
            num_chips_per_tile=40,
            image_transform=lambda x: torch.from_numpy(
                ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
            ).permute(2, 0, 1),
            label_transform=lambda y: torch.from_numpy(
                np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, y, mode="clip").astype(np.int64)
            ),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, num_workers=4, drop_last=False, pin_memory=True)


    # --- Loss and optimizer ---
    seg_loss_fn = HybridSegLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        ignore_index=0,
        ce_label_smoothing=args.ce_label_smoothing,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

#    # Freeze auxiliary branches
#    for p in model.recon_conv.parameters():
#        p.requires_grad = False
#    for p in model.aux_classifier.parameters():
#        p.requires_grad = False

    # --- Training loop ---
    for epoch in range(args.max_epochs):
        model.train()
        seg_losses = []

        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch}", leave=True)
        for imgs_norm, labs, _ in pbar:
            imgs_norm, labs = imgs_norm.to(device), labs.to(device).long()
            labs = torch.nan_to_num(labs, nan=0.0).clamp_(0, args.num_classes - 1)

            # Skip if all ignore
            if (labs != 0).sum() == 0:
                continue

            seg_logits, _, _ = model(imgs_norm)
            seg_logits = torch.nan_to_num(seg_logits, nan=0.0, posinf=1e4, neginf=-1e4)

            loss_seg = seg_loss_fn(seg_logits, labs)

            optimizer.zero_grad(set_to_none=True)
            loss_seg.backward()
            optimizer.step()

            seg_losses.append(loss_seg.item())
            pbar.set_postfix({"seg_loss": f"{np.mean(seg_losses):.4f}"})

        # --- Save checkpoint ---
        ckpt = os.path.join(args.savepath, f"seg_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"? Saved checkpoint {ckpt}")

        # --- Validation phase ---
        if val_loader is not None:
            validate_seg(model, val_loader, device, args.num_classes, epoch, args.savepath)


# =========================
# Validation function
# =========================
@torch.no_grad()
def validate_seg(model, val_loader, device, num_classes, epoch, save_dir):
    model.eval()
    y_true_all, y_pred_all = [], []

    for imgs_norm, labs, _ in tqdm(val_loader, desc="Validating", leave=False):
        imgs_norm, labs = imgs_norm.to(device), labs.to(device)
        seg_logits, _, _ = model(imgs_norm)
        preds = torch.argmax(seg_logits, dim=1)

        valid_mask = (labs != 0)
        y_true_all.append(labs[valid_mask].cpu().numpy().flatten())
        y_pred_all.append(preds[valid_mask].cpu().numpy().flatten())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    oa = accuracy_score(y_true_all, y_pred_all)
    ious = jaccard_score(y_true_all, y_pred_all, labels=list(range(num_classes)), average=None)
    miou = np.nanmean(ious)

    print("\n--- Validation Metrics ---")
    print(f"Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}")
    for cls_id, val in enumerate(ious):
        if cls_id in utils.LABEL_NAMES:
            cls_name = utils.LABEL_NAMES[cls_id]
        else:
            cls_name = f"Class {cls_id}"
        print(f"  {cls_name}: {val:.4f}")

    # --- Save to file ---
    os.makedirs(os.path.join(save_dir, "val_logs"), exist_ok=True)
    log_path = os.path.join(save_dir, "val_logs", "val_metrics.txt")
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}\n")
        for cls_id, val in enumerate(ious):
            f.write(f"  Class {cls_id}: {val:.4f}\n")
        f.write("\n")

    return miou

# =========================
# Utility: Sweep pseudo-label parameters
# =========================
def sweep_pseudo_params(args, model, device, thresh_vals, keep_vals):
    """
    Runs generate_pseudo_labels() for combinations of pseudo_thresh and pseudo_keep,
    logs how many pixels each class keeps, and saves summary CSV.
    """
    results = []
    df = pd.read_csv(args.list_dir)
    os.makedirs(os.path.join(args.savepath, "pseudo_sweeps"), exist_ok=True)

    for t in thresh_vals:
        for k in keep_vals:
            print(f"\n=== Sweep: pseudo_thresh={t:.2f}, pseudo_keep={k:.2f} ===")
            args.pseudo_thresh = t
            args.pseudo_keep = k

            # Temporary output directory for this combination
            combo_dir = os.path.join(args.savepath, f"pseudo_sweeps/t{t:.2f}_k{k:.2f}")
            os.makedirs(combo_dir, exist_ok=True)

            # Generate pseudo labels into combo_dir
            args.savepath = combo_dir
            generate_pseudo_labels(args, model, device)

            # After generation, compute pixel stats
            class_counts = np.zeros(args.num_classes, dtype=np.int64)
            total_pixels = 0
            for _, row in df.iterrows():
                fn = os.path.join(combo_dir, "pseudo_labels",
                                  os.path.basename(row["image_fn"]).replace(".tif", "_pseudo.tif"))
                if not os.path.exists(fn):
                    continue
                with rasterio.open(fn) as f:
                    data = f.read(1)
                    total_pixels += data.size
                    for c in range(args.num_classes):
                        class_counts[c] += np.sum(data == c)
            kept_pixels = int(class_counts.sum() - class_counts[0])
            kept_ratio = kept_pixels / total_pixels if total_pixels > 0 else 0

            # Log result
            result = {"pseudo_thresh": t, "pseudo_keep": k, "kept_ratio": kept_ratio}
            for c in range(args.num_classes):
                result[f"class_{c}_count"] = class_counts[c]
            results.append(result)

            print(f"  Kept {100*kept_ratio:.2f}% valid pixels.")
            print(f"  Class counts: {class_counts.tolist()}")

    # Save summary
    out_csv = os.path.join(args.savepath, "pseudo_sweep_summary.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n✅ Sweep summary saved to {out_csv}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Chesapeake")
    parser.add_argument("--list_dir", type=str, default="./dataset/CSV_list/Chesapeake_NewYork.csv")

#    parser.add_argument("--val_list", type=str, default="./dataset/CSV_list/C_NYC-val.csv",
#                    help="CSV file for validation set (with HR_label_fns column).")

    parser.add_argument("--max_epochs", type=int, default=31)
    parser.add_argument("--epochs_recon", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--base_lr", type=float, default=0.003) #v1-lr_0.003 (original) v2-lr_0.0001
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--savepath", type=str, default="./log_l2_qunet/test/exp/")#"./log_l2_qunet/ablation/self-masked_pseudo/")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--chip_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_chips_per_tile", type=int, default=100)
    
    # Loss/strategy knobs
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--freeze_recon_epoch", type=int, default=1, help="epoch at which recon head is frozen")
    parser.add_argument("--aux_weight_after_freeze", type=float, default=2.0, help="boost aux training after freeze")
    parser.add_argument("--aux_label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    
    # Seg loss
    parser.add_argument("--ce_weight", type=float, default=0.6) #original- 0.7
    parser.add_argument("--dice_weight", type=float, default=0.4) #original - 0.3

    parser.add_argument("--ce_label_smoothing", type=float, default=0.02)


    
    # Pseudo label controls
    parser.add_argument("--pseudo_thresh", type=float, default=0.25) #0.3-v1, 0.25-v2
    parser.add_argument("--pseudo_keep", type=float, default=0.25, #original 0.2
                        help="if >0, keep this FRACTION of most-confident pixels (within valid mask) instead of absolute threshold")
    parser.add_argument("--aux_temp", type=float, default=0.7, help="temperature for pseudo logits; <1 sharpens") #0.7 original
    parser.add_argument("--morph_open", type=int, default=1, help="morphological opening iterations for denoise")
    
    #parser.add_argument("--phase", type=str, choices=["recon", "pseudo", "seg"], required=True)
    parser.add_argument("--phase", type=str, choices=["recon", "pseudo", "seg", "pseudo_sweep"], required=True)

    
    args = parser.parse_args()

#    if torch.cuda.is_available():
#        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#        print(f"Using CUDA device(s): {args.gpu}")
#    else:
#        print("Using CPU")
    if torch.cuda.is_available():
        print(f"Using CUDA device(s): {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    os.makedirs(args.savepath, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QuantumUNet(num_classes=args.num_classes, in_channels=4).to(device)
    summary(net)

    if args.phase == "recon":
        train_recon(args, net, device)
    elif args.phase == "pseudo":
        ckpt = os.path.join(args.savepath, f"recon_epoch_{args.epochs_recon-1}.pth")
        state_dict = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(state_dict); net = net.to(device)
        summary(net)
        generate_pseudo_labels(args, net, device)
        
    elif args.phase == "pseudo_sweep":
        ckpt = os.path.join(args.savepath, f"recon_epoch_{args.epochs_recon-1}.pth")
        state_dict = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(state_dict); net = net.to(device)
        # run sweep
        sweep_pseudo_params(
            args, net, device,
            thresh_vals=[0.25, 0.3, 0.35],
            keep_vals=[0.15, 0.20, 0.25]
        )
    elif args.phase == "seg":
        ckpt = os.path.join(args.savepath, f"recon_epoch_{args.epochs_recon-1}.pth")
        net.load_state_dict(torch.load(ckpt, map_location=device))
        summary(net)
        train_seg(args, net, device)


if __name__ == "__main__":
    main()

