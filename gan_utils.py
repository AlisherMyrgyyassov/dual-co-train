"""
GAN utilities: conditional Pix2Pix generator/discriminator, pretraining, and fine-tuning.

All model definitions, training loops, and helper functions for the conditional GAN
used in the dual co-training pipeline.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks.unet import UNet


# ---------------------------------------------------------------------------
# Perceptual loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=(3, 8, 15, 22), device="cpu"):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.feature_layers = feature_layers
        self.criterion = nn.L1Loss()
        self.device = device

    def forward(self, generated, real):
        if generated.shape[1] == 1:
            generated_rgb = generated.repeat(1, 3, 1, 1)
            real_rgb = real.repeat(1, 3, 1, 1)
        else:
            generated_rgb = generated
            real_rgb = real
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        generated_norm = (generated_rgb - mean) / std
        real_norm = (real_rgb - mean) / std
        gen_feats, real_feats = [], []
        x_gen, x_real = generated_norm, real_norm
        for i, layer in enumerate(self.vgg):
            x_gen = layer(x_gen)
            x_real = layer(x_real)
            if i in self.feature_layers:
                gen_feats.append(x_gen)
                real_feats.append(x_real)
        loss = sum(self.criterion(g, r) for g, r in zip(gen_feats, real_feats))
        return loss / len(gen_feats)


# ---------------------------------------------------------------------------
# PatchGAN discriminator
# ---------------------------------------------------------------------------

class PatchGAN(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        def conv_block(in_c, out_c, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 1)

    def forward(self, mask, image, noise):
        x = torch.cat([mask, image, noise], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)


# ---------------------------------------------------------------------------
# Dataset (image–mask pairs)
# ---------------------------------------------------------------------------

class Pix2PixDataset(Dataset):
    def __init__(
        self,
        images_folder,
        masks_folder=None,
        noise_amplitude=0.1,
        seed=42,
        masks_by_stem: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.masks_by_stem = dict(masks_by_stem) if masks_by_stem is not None else None
        self.noise_amplitude = float(noise_amplitude)
        self.seed = int(seed)

        if (self.masks_by_stem is None) == (self.masks_folder is None):
            raise ValueError("Pix2PixDataset: pass exactly one of masks_folder or masks_by_stem")

        self.image_files = sorted(f for f in os.listdir(images_folder) if f.endswith(".png"))
        self.matched_pairs = []
        if self.masks_by_stem is not None:
            for img_file in self.image_files:
                stem = os.path.splitext(img_file)[0]
                if stem in self.masks_by_stem:
                    self.matched_pairs.append((img_file, stem))
        else:
            mask_files = sorted(f for f in os.listdir(self.masks_folder) if f.endswith(".png"))
            for img_file in self.image_files:
                stem = os.path.splitext(img_file)[0]
                mf = f"{stem}.mask.png"
                if mf in mask_files:
                    self.matched_pairs.append((img_file, mf))

        if not self.matched_pairs:
            raise ValueError("No matching image-mask pairs found!")

        first_img = Image.open(os.path.join(self.images_folder, self.matched_pairs[0][0])).convert("L")
        w, h = first_img.size
        g = torch.Generator().manual_seed(self.seed)
        self.noise_maps = [
            torch.randn((1, h, w), generator=g) * self.noise_amplitude
            for _ in range(len(self.matched_pairs))
        ]

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        img_file, mask_key = self.matched_pairs[idx]
        image = np.array(Image.open(os.path.join(self.images_folder, img_file)).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image[None, ...]).float()
        h, w = image.shape[1:]

        if self.masks_by_stem is not None:
            u8 = np.asarray(self.masks_by_stem[mask_key], dtype=np.uint8)
            mask = (u8.astype(np.float32) / 255.0)[None, ...]
        else:
            mask_img = Image.open(os.path.join(self.masks_folder, mask_key)).convert("L")
            mask = (np.array(mask_img, dtype=np.float32) / 255.0)[None, ...]
        mask = torch.from_numpy((mask > 0.5).astype(np.float32)).float()

        noise = self.noise_maps[idx].clone()
        input_cond = torch.cat([mask, noise], dim=0)
        return input_cond, image


# ---------------------------------------------------------------------------
# Weights init
# ---------------------------------------------------------------------------

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def add_noise(tensor, std=0.1):
    return tensor + torch.randn_like(tensor) * std


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------

def _build_preview_batch(dataset, num_samples: int, seed: int, device: torch.device):
    n_total = len(dataset)
    n = max(1, min(num_samples, n_total))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_total, size=n, replace=False)) if n < n_total else np.arange(n_total)
    xs, ys = [], []
    for idx in chosen.tolist():
        x, y = dataset[int(idx)]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0).to(device), torch.stack(ys, dim=0).to(device), chosen.tolist()


@torch.no_grad()
def _save_preview_grid(generator, input_cond, real_images, out_path, epoch_num):
    generator.eval()
    masks = input_cond[:, 0:1]
    fake = torch.sigmoid(generator(input_cond)).detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    real_np = real_images.detach().cpu().numpy()
    n = masks_np.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, max(2.5, n * 2.0)))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for c, t in enumerate(["Mask", "Generated", "Real"]):
        axes[0, c].set_title(t)
    for r in range(n):
        axes[r, 0].imshow(masks_np[r, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(fake[r, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(real_np[r, 0], cmap="gray", vmin=0, vmax=1)
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle(f"Epoch {epoch_num:03d}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# GAN training loop (pretrain)
# ---------------------------------------------------------------------------

def train_gan(
    generator,
    discriminator,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate_g,
    learning_rate_d,
    device,
    lambda_l1=10,
    lambda_perceptual=8,
    lambda_gan=1.0,
    checkpoint_dir="gan-checkpoints",
    save_every=10,
    label_smoothing=0.9,
    noise_std=0.02,
    d_train_full_epochs=3,
    d_update_every=2,
    preview_every=5,
    preview_num_samples=8,
    preview_seed=42,
):
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss(device=device)

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))

    scheduler_g = optim.lr_scheduler.LambdaLR(
        optimizer_g, lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / (num_epochs // 2)
    )
    scheduler_d = optim.lr_scheduler.LambdaLR(
        optimizer_d, lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / (num_epochs // 2)
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    g_losses, d_losses = [], []
    best_val_loss = float("inf")

    real_label = label_smoothing
    fake_label = 0.0
    d_update_every = max(1, int(d_update_every))

    preview_cond = None
    if preview_every > 0:
        try:
            preview_cond, preview_real, _ = _build_preview_batch(
                val_loader.dataset, preview_num_samples, preview_seed, device
            )
        except Exception:
            preview_every = 0

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (input_cond, real_images) in enumerate(pbar):
            input_cond = input_cond.to(device)
            real_images = real_images.to(device)
            mask = input_cond[:, 0:1]
            noise = input_cond[:, 1:2]
            fake_images = torch.sigmoid(generator(input_cond))

            train_discriminator = (epoch < d_train_full_epochs) or ((batch_idx % d_update_every) == 0)

            if train_discriminator:
                optimizer_d.zero_grad()
                pred_real = discriminator(mask, add_noise(real_images, noise_std), noise)
                loss_d_real = criterion_gan(
                    pred_real,
                    torch.full_like(pred_real, real_label, device=device),
                )
                pred_fake = discriminator(mask, add_noise(fake_images.detach(), noise_std), noise)
                loss_d_fake = criterion_gan(
                    pred_fake,
                    torch.full_like(pred_fake, fake_label, device=device),
                )
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                optimizer_d.step()
            else:
                with torch.no_grad():
                    pred_real = discriminator(mask, add_noise(real_images, noise_std), noise)
                    loss_d_real = criterion_gan(
                        pred_real,
                        torch.full_like(pred_real, real_label, device=device),
                    )
                    pred_fake = discriminator(mask, add_noise(fake_images.detach(), noise_std), noise)
                    loss_d_fake = criterion_gan(
                        pred_fake,
                        torch.full_like(pred_fake, fake_label, device=device),
                    )
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

            optimizer_g.zero_grad()
            pred_fake = discriminator(mask, add_noise(fake_images, noise_std), noise)
            loss_g_gan = criterion_gan(
                pred_fake,
                torch.full_like(pred_fake, real_label, device=device),
            )
            loss_g_l1 = criterion_l1(fake_images, real_images) * lambda_l1
            loss_g_perceptual = criterion_perceptual(fake_images, real_images) * lambda_perceptual
            loss_g = lambda_gan * loss_g_gan + loss_g_l1 + loss_g_perceptual
            loss_g.backward()
            optimizer_g.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                pbar.set_postfix(
                    G_loss=f"{loss_g.item():.4f}",
                    D_loss=f"{loss_d.item():.4f}",
                    G_l1=f"{loss_g_l1.item():.4f}",
                    G_perc=f"{loss_g_perceptual.item():.4f}",
                )

        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        val_loss = evaluate_gan(generator, val_loader, criterion_l1, device)
        scheduler_g.step()
        scheduler_d.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | "
            f"Val L1: {val_loss:.4f} | LR: {optimizer_g.param_groups[0]['lr']:.6f}"
        )

        if preview_every > 0 and (epoch + 1) % preview_every == 0 and preview_cond is not None:
            try:
                _save_preview_grid(
                    generator, preview_cond, preview_real,
                    os.path.join(checkpoint_dir, "visualizations", f"epoch_{epoch+1:03d}_samples.png"),
                    epoch + 1,
                )
            except Exception:
                pass

        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "g_losses": g_losses,
                    "d_losses": d_losses,
                    "val_loss": val_loss,
                },
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(checkpoint_dir, "best_model.pth"),
            )

    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(
        {
            "epoch": num_epochs,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_losses": g_losses,
            "d_losses": d_losses,
            "val_loss": val_loss,
        },
        final_path,
    )
    return g_losses, d_losses


@torch.no_grad()
def evaluate_gan(generator, data_loader, criterion, device):
    generator.eval()
    total = 0.0
    n = 0
    for input_cond, real_images in data_loader:
        input_cond = input_cond.to(device)
        real_images = real_images.to(device)
        fake = torch.sigmoid(generator(input_cond))
        total += criterion(fake, real_images).item()
        n += 1
    return total / max(1, n)


def _load_gan_checkpoint_for_finetune(ckpt_path, generator, discriminator, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")
    gen_state = ckpt.get("generator_state_dict")
    if gen_state is None:
        raise ValueError(f"Checkpoint missing generator_state_dict: {ckpt_path}")
    gen_state = {k.replace("module.", ""): v for k, v in gen_state.items()}
    generator.load_state_dict(gen_state, strict=False)
    disc_state = ckpt.get("discriminator_state_dict")
    if isinstance(disc_state, dict):
        disc_state = {k.replace("module.", ""): v for k, v in disc_state.items()}
        discriminator.load_state_dict(disc_state, strict=False)


# ---------------------------------------------------------------------------
# GAN fine-tuning (used during co-training)
# ---------------------------------------------------------------------------

def finetune_gan(
    *,
    ckpt_in,
    images_folder,
    masks_folder=None,
    checkpoint_dir,
    device: torch.device,
    num_epochs: int,
    batch_size: int = 8,
    num_workers: int = 4,
    learning_rate_g: float = 1e-4,
    learning_rate_d: float = 1e-4,
    lambda_l1: float = 10,
    lambda_perceptual: float = 20,
    lambda_gan: float = 1.0,
    label_smoothing: float = 0.9,
    noise_std: float = 0.05,
    noise_amplitude: float = 0.1,
    seed: int = 42,
    d_train_full_epochs: int = 3,
    d_update_every: int = 2,
    preview_every: int = 5,
    preview_num_samples: int = 8,
    preview_seed: int = 42,
    masks_by_stem=None,
) -> Path:
    """Fine-tune the conditional GAN from an existing checkpoint using latest pseudo-masks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if (masks_folder is None) == (masks_by_stem is None):
        raise ValueError("finetune_gan: pass exactly one of masks_folder or masks_by_stem")

    if masks_by_stem is not None:
        full_dataset = Pix2PixDataset(
            images_folder=images_folder, masks_folder=None,
            noise_amplitude=noise_amplitude, seed=seed, masks_by_stem=masks_by_stem,
        )
    else:
        full_dataset = Pix2PixDataset(
            images_folder=images_folder, masks_folder=str(masks_folder),
            noise_amplitude=noise_amplitude, seed=seed,
        )

    ds_size = len(full_dataset)
    val_size = max(1, int(ds_size * 0.15))
    train_size = ds_size - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))

    generator = UNet(n_channels=2, n_classes=1).to(device)
    discriminator = PatchGAN(input_channels=3).to(device)
    _load_gan_checkpoint_for_finetune(ckpt_in, generator, discriminator, device)

    train_gan(
        generator=generator, discriminator=discriminator,
        train_loader=train_loader, val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d,
        device=device,
        lambda_l1=lambda_l1, lambda_perceptual=lambda_perceptual, lambda_gan=lambda_gan,
        checkpoint_dir=checkpoint_dir, save_every=5,
        label_smoothing=label_smoothing, noise_std=noise_std,
        d_train_full_epochs=d_train_full_epochs, d_update_every=d_update_every,
        preview_every=preview_every, preview_num_samples=preview_num_samples, preview_seed=preview_seed,
    )
    return Path(checkpoint_dir) / "best_model.pth"


# ---------------------------------------------------------------------------
# GAN pretrain (from scratch)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GANConfig:
    images_folder: str = ""
    masks_folder: str = ""
    ultraunet_mask_checkpoint: Optional[str] = None
    seg_mask_threshold: float = 0.5
    checkpoint_dir: str = "checkpoints/gan"
    batch_size: int = 8
    epochs: int = 30
    num_workers: int = 4
    seed: int = 42
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    lambda_l1: float = 10.0
    lambda_perceptual: float = 8.0
    lambda_gan: float = 1.0
    label_smoothing: float = 0.9
    noise_std: float = 0.02
    noise_amplitude: float = 0.1
    d_train_full_epochs: int = 3
    d_update_every: int = 2
    save_every: int = 5
    preview_every: int = 5
    preview_num_samples: int = 8
    preview_seed: int = 42


def pretrain_gan(config=None, *, device=None, **overrides) -> dict:
    """Pretrain a conditional GAN from scratch on image-mask pairs."""
    if config is None:
        config = GANConfig()
    if overrides:
        cfg_dict = asdict(config)
        unknown = sorted(k for k in overrides if k not in cfg_dict)
        if unknown:
            raise TypeError(f"Unknown config override(s): {unknown}")
        cfg_dict.update(overrides)
        config = GANConfig(**cfg_dict)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    masks_by_stem = None
    umc = config.ultraunet_mask_checkpoint
    if umc:
        from utils.pseudo_mask_infer import infer_pseudo_masks_u8_dict, iter_image_stems, load_ultraunet_from_checkpoint
        ckpt_path = Path(umc)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"ultraunet_mask_checkpoint not found: {ckpt_path}")
        seg_model = load_ultraunet_from_checkpoint(ckpt_path, device=device)
        stems_inf = iter_image_stems(Path(config.images_folder))
        masks_by_stem = infer_pseudo_masks_u8_dict(seg_model, Path(config.images_folder), stems_inf,
                                                     config.seg_mask_threshold, device)

    full_dataset = Pix2PixDataset(
        images_folder=config.images_folder,
        masks_folder=None if masks_by_stem is not None else config.masks_folder,
        noise_amplitude=config.noise_amplitude,
        seed=config.seed,
        masks_by_stem=masks_by_stem,
    )

    ds_size = len(full_dataset)
    val_size = max(1, int(ds_size * 0.15))
    train_size = ds_size - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(config.seed))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                               num_workers=config.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=(device.type == "cuda"))

    generator = UNet(n_channels=2, n_classes=1).to(device)
    discriminator = PatchGAN(input_channels=3).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    g_losses, d_losses = train_gan(
        generator=generator, discriminator=discriminator,
        train_loader=train_loader, val_loader=val_loader,
        num_epochs=config.epochs,
        learning_rate_g=config.lr_g, learning_rate_d=config.lr_d,
        device=device,
        lambda_l1=config.lambda_l1, lambda_perceptual=config.lambda_perceptual,
        lambda_gan=config.lambda_gan,
        checkpoint_dir=config.checkpoint_dir, save_every=config.save_every,
        label_smoothing=config.label_smoothing, noise_std=config.noise_std,
        d_train_full_epochs=config.d_train_full_epochs, d_update_every=config.d_update_every,
        preview_every=config.preview_every, preview_num_samples=config.preview_num_samples,
        preview_seed=config.preview_seed,
    )

    return {
        "config": config,
        "device": device,
        "g_losses": g_losses,
        "d_losses": d_losses,
        "checkpoint_dir": Path(config.checkpoint_dir),
        "best_model_path": Path(config.checkpoint_dir) / "best_model.pth",
        "final_model_path": Path(config.checkpoint_dir) / "final_model.pth",
    }
