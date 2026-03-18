#!/usr/bin/env python3
"""
main_soa.py
Train segmentation models from scratch (UNet / DeepLabV3 / SegFormer / TransUNet)
using streaming tiles (same style as your original main.py).

Example:
python main_soa.py \
    --list_dir ./dataset/CSV_list/Chesapeake_NewYork.csv \
    --val_list ./dataset/CSV_list/C_NYC-val.csv \
    --savepath ./runs_from_scratch/unet_exp \
    --model unet \
    --num_classes 6 \
    --in_channels 4 \
    --chip_size 256 \
    --max_epochs 30
"""
import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

import utils
from networks.soa import get_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchinfo import summary

# ------------------------
# Streaming dataset (same pattern as your main.py)
# ------------------------
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

                height, width = img_fp.height, img_fp.width
                chips_yielded = 0

                while chips_yielded < self.num_chips_per_tile:
                    if width <= self.chip_size or height <= self.chip_size:
                        x, y = 0, 0
                    else:
                        x = np.random.randint(0, width - self.chip_size)
                        y = np.random.randint(0, height - self.chip_size)

                    img = np.rollaxis(
                        img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3
                    )

                    if self.image_transform is not None:
                        img_norm = self.image_transform(img)
                    else:
                        img_norm = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0))

                    img_raw = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

                    lr_labels = None
                    if lr_fp is not None:
                        lr_labels = lr_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                        if self.label_transform is not None:
                            lr_labels = self.label_transform(lr_labels)

                    hr_labels = None
                    if hr_fp is not None:
                        hr_labels = hr_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                        if self.label_transform is not None:
                            hr_labels = self.label_transform(hr_labels)

                    label_to_use = hr_labels if hr_labels is not None else lr_labels
                    if label_to_use is not None:
                        yield img_norm, label_to_use, img_raw
                    else:
                        yield img_norm, img_raw

                    chips_yielded += 1


# ------------------------
# Loss utilities (HybridSegLoss from your main.py)
# ------------------------
def weighted_dice_loss(inputs, targets, ignore_index=0, eps=1e-6):
    C = inputs.shape[1]
    probs = F.softmax(inputs, dim=1)
    valid = (targets != ignore_index)
    if valid.sum() == 0:
        return inputs.new_tensor(0.0)
    onehot = F.one_hot(targets.clamp(min=0), num_classes=C).permute(0, 3, 1, 2).float()
    probs = probs * valid.unsqueeze(1)
    onehot = onehot * valid.unsqueeze(1)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * onehot, dims)
    cardinality  = torch.sum(probs + onehot, dims)
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
        try:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=ce_label_smoothing)
        except TypeError:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e4, neginf=-1e4)
        targets = torch.clamp(targets, 0, inputs.shape[1]-1).long()
        ce_val = self.ce_loss(inputs, targets)
        dice_val = weighted_dice_loss(inputs, targets, ignore_index=self.ignore_index)
        loss = self.ce_weight * ce_val + self.dice_weight * dice_val
        if torch.isnan(loss) or torch.isinf(loss):
            return inputs.new_tensor(0.0)
        return loss


# ------------------------
# Validation (tile-chips streaming)
# ------------------------
from sklearn.metrics import jaccard_score, accuracy_score, cohen_kappa_score

@torch.no_grad()
def validate_seg(model, val_loader, device, num_classes, epoch, save_dir):
    model.eval()
    y_true_all, y_pred_all = [], []

    for imgs_norm, labs, _ in tqdm(val_loader, desc="Validating", leave=False):
        imgs_norm, labs = imgs_norm.to(device), labs.to(device)
        out = model(imgs_norm)
        # unwrap possible outputs
        if isinstance(out, (tuple, list)):
            seg_logits = out[0]
        elif isinstance(out, dict):
            seg_logits = out.get("logits", list(out.values())[0])
        else:
            seg_logits = out

        preds = torch.argmax(seg_logits, dim=1)
        valid_mask = (labs != 0)
        if valid_mask.sum() == 0:
            continue
        y_true_all.append(labs[valid_mask].cpu().numpy().flatten())
        y_pred_all.append(preds[valid_mask].cpu().numpy().flatten())

    if len(y_true_all) == 0:
        print("[Warning] No valid pixels in validation set.")
        return 0.0

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    oa = accuracy_score(y_true_all, y_pred_all)
    ious = jaccard_score(y_true_all, y_pred_all, labels=list(range(num_classes)), average=None, zero_division=0)
    miou = np.nanmean(ious)
    kappa = cohen_kappa_score(y_true_all, y_pred_all)

    print(f"[Val] Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}, Kappa={kappa:.4f}")

    os.makedirs(os.path.join(save_dir, "val_logs"), exist_ok=True)
    log_path = os.path.join(save_dir, "val_logs", "val_metrics.txt")
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}, Kappa={kappa:.4f}\n")

    return miou


# ------------------------
# Train function (streaming dataset)
# ------------------------
def train_seg(args, model, device):
    # prepare train dataset from CSV (train uses image_fn,label_fn)
    df = pd.read_csv(args.list_dir)
    train_dataset = StreamingGeospatialDataset(
        imagery_fns=df["image_fn"].values,
        lr_label_fns=df["label_fn"].values,
        chip_size=args.chip_size,
        num_chips_per_tile=args.num_chips_per_tile,
        image_transform=lambda x: torch.from_numpy(
            ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
        ).permute(2, 0, 1),
        label_transform=lambda y: torch.from_numpy(
            np.take(utils.LABEL_CLASS_TO_IDX_MAP, y, mode="clip").astype(np.int64)
        ),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              drop_last=True, pin_memory=True, prefetch_factor=4)


    # validation dataset from val_list (image_fn, hr_label_fn)
    val_loader = None
    if args.val_list and os.path.exists(args.val_list):
        df_val = pd.read_csv(args.val_list)
        if "hr_label_fn" not in df_val.columns:
            raise KeyError("Validation CSV must include 'hr_label_fn' column.")
        val_dataset = StreamingGeospatialDataset(
            imagery_fns=df_val["image_fn"].values,
            hr_label_fns=df_val["hr_label_fn"].values,
            chip_size=args.chip_size,
            num_chips_per_tile=args.num_chips_per_tile_val,
            image_transform=lambda x: torch.from_numpy(
                ((x - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
            ).permute(2, 0, 1),
            label_transform=lambda y: torch.from_numpy(
                np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, y, mode="clip").astype(np.int64)
            ),
        )
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=4, drop_last=False, pin_memory=True)

    seg_loss_fn = HybridSegLoss(ce_weight=args.ce_weight, dice_weight=args.dice_weight,
                                ignore_index=0, ce_label_smoothing=args.ce_label_smoothing)

#    seg_loss_fn = nn.CrossEntropyLoss()

    
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        print(f"[Info] Using DataParallel on {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    best_miou = -1.0
    os.makedirs(args.savepath, exist_ok=True)

    for epoch in range(args.max_epochs):
        model.train()
        seg_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for imgs_norm, labs, _ in pbar:
            imgs_norm = imgs_norm.to(device)
            labs = labs.to(device).long()
            labs = torch.nan_to_num(labs, nan=0.0).clamp_(0, args.num_classes - 1)

            if (labs != 0).sum() == 0:
                continue

            out = model(imgs_norm)
            if isinstance(out, (tuple, list)):
                seg_logits = out[0]
            elif isinstance(out, dict):
                seg_logits = out.get("logits", list(out.values())[0])
            else:
                seg_logits = out

            seg_logits = torch.nan_to_num(seg_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_seg = seg_loss_fn(seg_logits, labs)

            optimizer.zero_grad(set_to_none=True)
            loss_seg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            seg_losses.append(loss_seg.item())
            pbar.set_postfix({"seg_loss": f"{np.mean(seg_losses):.4f}"})

        # save checkpoint
        ckpt = os.path.join(args.savepath, f"{args.model}_epoch_{epoch}.pth")
        torch.save(model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")

        if val_loader is not None:
            miou = validate_seg(model, val_loader, device, args.num_classes, epoch, args.savepath)
            if miou > best_miou:
                best_miou = miou
                best_ckpt = os.path.join(args.savepath, f"{args.model}_best.pth")
                torch.save(model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(), best_ckpt)
                print(f"Saved BEST checkpoint: {best_ckpt} (mIoU={miou:.4f})")

    print("Training finished.")


# ------------------------
# CLI
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_dir", type=str, default="./dataset/CSV_list/Chesapeake_NewYork.csv")
    parser.add_argument("--val_list", type=str, default="./dataset/CSV_list/C_NYC-val.csv")
    parser.add_argument("--savepath", type=str, default="./log_l2_qunet/soa/quantumunet")
    parser.add_argument("--model", type=str, default="quantumunet", choices=["unet","deeplabv3","segformer","transunet", "quantumunet"])
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--chip_size", type=int, default=256)
    parser.add_argument("--num_chips_per_tile", type=int, default=100)
    parser.add_argument("--num_chips_per_tile_val", type=int, default=40)
    parser.add_argument("--max_epochs", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=0.003)
    parser.add_argument("--ce_weight", type=float, default=0.6)
    parser.add_argument("--dice_weight", type=float, default=0.4)
    parser.add_argument("--ce_label_smoothing", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_dataparallel", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()

def prepare_model(args, device):
    print(f"Building model '{args.model}' (in_channels={args.in_channels}, num_classes={args.num_classes})")
    model = get_model(args.model, num_classes=args.num_classes, in_channels=args.in_channels, img_size=args.chip_size)
    model = model.to(device)
    return model

if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
        print(f"Using CUDA device(s): {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    os.makedirs(args.savepath, exist_ok=True)
    model = prepare_model(args, device)
    summary(model)
    train_seg(args, model, device)
