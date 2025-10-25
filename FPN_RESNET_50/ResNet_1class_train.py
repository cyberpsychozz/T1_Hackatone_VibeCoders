# train_fpn_resnet50_cihp.py
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Импорт из репозитория pytorch-fpn
from fpn.factory import make_fpn_resnet


# -----------------------
# Dataset
# -----------------------
class CIHPDataset(Dataset):
    def __init__(self, root: str, list_file: str, img_size=384, augment=True, mode="train"):
        self.root = Path(root)
        self.mode = mode  # train, val, or test
        self.images_dir = self.root / mode / "Images"
        self.masks_dir = self.root / mode / "Category_ids"  # Using binary masks from Category_ids
        with open(list_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        self.ids = lines
        self.img_size = img_size
        self.augment = augment
        self.train_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
            A.RandomGamma(gamma_limit=(60,140), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(img_size, img_size),
        ])
        self.val_transform = A.Compose([A.Resize(img_size, img_size)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = self.images_dir / f"{id_}.jpg"
        mask_path = self.masks_dir / f"{id_}_seg.png"  # Assuming binary masks have '_seg.png' suffix
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = (mask > 0).astype('uint8')  # Binary mask: 0 for background, 1 for human

        if self.augment:
            aug = self.train_transform(image=img, mask=mask)
        else:
            aug = self.val_transform(image=img, mask=mask)
        img, mask = aug['image'], aug['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return {"image": img, "mask": mask}


# -----------------------
# Losses
# -----------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        intersection = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)
        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


# -----------------------
# Helpers
# -----------------------
def compute_miou(preds: np.ndarray, target: np.ndarray):
    return jaccard_score(target.flatten(), preds.flatten(), average='binary')


# -----------------------
# Training loop
# -----------------------
def train_loop(
    dataset_root: str,
    train_list: str,
    val_list: str,
    output_dir: str = "outputs",
    image_size: int = 384,
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 6e-5,
    device: str = "cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    train_ds = CIHPDataset(dataset_root, train_list, img_size=image_size, augment=True, mode="Training")
    val_ds = CIHPDataset(dataset_root, val_list, img_size=image_size, augment=False, mode="Validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize FPN + ResNet50 model from the cloned repository
    model = make_fpn_resnet(
        name='resnet50',  # ResNet50 backbone
        fpn_type='fpn',   # Vanilla FPN
        pretrained=True,  # Use pretrained weights
        num_classes=1,    # Binary segmentation (1 channel for foreground logits)
        fpn_channels=256, # Default FPN channels
        in_channels=3,    # RGB input
        out_size=(image_size, image_size)  # Output size matching input
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    scaler = torch.amp.GradScaler()

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    best_miou = 0.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train")
        running_loss = 0.0
        for batch in pbar:
            imgs = batch['image'].to(device)
            masks = batch['mask'].unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                # Forward pass: model outputs logits [B, 1, H, W]
                logits_bin = model(imgs)

                # Resize masks to match logits size if necessary (FPN typically matches input size)
                if logits_bin.shape[2:] != masks.shape[2:]:
                    masks_resized = torch.nn.functional.interpolate(masks, size=logits_bin.shape[2:], mode='nearest')
                else:
                    masks_resized = masks

                loss_bce = bce(logits_bin, masks_resized)
                loss_dice = dice(logits_bin, masks_resized)
                loss = 0.6*loss_bce + 0.4*loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': running_loss/(global_step if global_step>0 else 1)})

        # Validation
        model.eval()
        miou_scores = []
        with torch.no_grad():
            for vb in tqdm(val_loader, desc="Validation"):
                img = vb['image'].to(device)
                mask = vb['mask'].cpu().numpy().astype(np.uint8)

                with torch.amp.autocast(device_type=device.type):
                    out = model(img)
                logits_bin = out  # [1, 1, H, W]

                # Resize logits to original mask size if necessary
                if logits_bin.shape[2:] != mask.shape[1:]:
                    probs = torch.sigmoid(logits_bin)
                    probs_resized = torch.nn.functional.interpolate(probs, size=mask.shape[1:], mode='bilinear', align_corners=False)[0,0].cpu().numpy()
                else:
                    probs = torch.sigmoid(logits_bin)
                    probs_resized = probs[0,0].cpu().numpy()

                pred_mask = (probs_resized > 0.5).astype(np.uint8)
                miou_scores.append(compute_miou(pred_mask, mask))

        avg_miou = float(np.mean(miou_scores))
        print(f"Epoch {epoch+1} val mIoU: {avg_miou:.4f}")

        # Save checkpoints
        ckpt_path = os.path.join(output_dir, f"ckpt_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, ckpt_path)

        if avg_miou > best_miou:
            best_miou = avg_miou
            best_path = os.path.join(output_dir, "best.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_path)
            print("Saved best:", best_path)

    print("Training done. Best mIoU:", best_miou)


# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="dataset_CIHP")
    parser.add_argument("--train-list", type=str, default="dataset_CIHP/Training/train_id.txt")
    parser.add_argument("--val-list", type=str, default="dataset_CIHP/Validation/val_id.txt")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_loop(
        dataset_root=args.dataset_root,
        train_list=args.train_list,
        val_list=args.val_list,
        output_dir=args.out,
        image_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )