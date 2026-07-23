"""
Training loop and evaluation harness for the U-Net baseline (Phase 4).

Handles:
- Training with checkpointing (resumable after Colab disconnects)
- Evaluation: MRE, SDR @ 2/2.5/3/4mm, bootstrap 95% CIs, per-landmark breakdown
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aariz_dataset import AarizDataset, decode_heatmaps, pixel_error_to_mm, LANDMARK_ORDER
from unet_model import UNet


class WeightedHeatmapMSE(nn.Module):
    """Plain MSE on heatmaps lets the optimizer 'cheat' by predicting near-zero everywhere —
    background pixels vastly outnumber the small Gaussian peak (e.g. sigma=2 on a 128x128
    heatmap: the peak region is a tiny fraction of total pixels), so all-zero output already
    achieves a very low loss without learning anything. This was confirmed empirically: a
    real training run's loss plateaued at ~0.0008, matching the theoretical MSE of an all-zero
    prediction (~0.00077 for sigma=2) almost exactly — the model had collapsed to that trivial
    solution instead of learning to localize landmarks (SDR@2mm was ~0%).

    Fix: weight each pixel's squared error by (1 + pos_weight * heatmap_gt), so pixels near
    the true landmark location contribute far more to the loss than background pixels do.
    This is standard practice in heatmap-regression literature for exactly this failure mode.
    """
    def __init__(self, pos_weight=100.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        weight = 1.0 + self.pos_weight * target
        return (weight * (pred - target) ** 2).mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        heatmaps_gt = batch["heatmaps"].to(device)

        optimizer.zero_grad()
        heatmaps_pred = model(images)
        loss = criterion(heatmaps_pred, heatmaps_gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, dataset, device, image_size=512, heatmap_size=256, batch_size=8):
    """Returns a dict with per-sample, per-landmark mm errors (N_samples, 29), plus
    dataset-level MRE/SDR summary. Keep the raw per-sample array around — it's what
    bootstrap_ci and per_landmark_table both need, and recomputing it twice would be wasteful."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors_mm = []  # will become (N, 29)
    for batch in loader:
        images = batch["image"].to(device)
        heatmaps_pred = model(images).cpu()
        gt = batch["landmarks_px_resized"].numpy()  # (B, 29, 2)

        for i in range(images.size(0)):
            decoded = decode_heatmaps(heatmaps_pred[i], image_size, heatmap_size)
            orig_size = (batch["orig_size"][0][i].item(), batch["orig_size"][1][i].item())
            pixel_size_mm = batch["pixel_size_mm"][i].item()
            err_mm = pixel_error_to_mm(decoded, gt[i], orig_size, image_size, pixel_size_mm)
            all_errors_mm.append(err_mm)

    all_errors_mm = np.stack(all_errors_mm, axis=0)  # (N, 29)
    return all_errors_mm


@torch.no_grad()
def evaluate_with_devices(model, dataset, device, image_size=512, heatmap_size=256, batch_size=8):
    """Same as evaluate(), but also returns the imaging device/machine name for each sample,
    so results can be grouped by device afterward (Phase 7 — is performance uniform across
    Aariz's 7 imaging devices, or concentrated in specific ones?).
    Requires the dataset's __getitem__ to include a 'machine' field (AarizDataset does)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors_mm = []
    all_machines = []
    for batch in loader:
        images = batch["image"].to(device)
        heatmaps_pred = model(images).cpu()
        gt = batch["landmarks_px_resized"].numpy()

        for i in range(images.size(0)):
            decoded = decode_heatmaps(heatmaps_pred[i], image_size, heatmap_size)
            orig_size = (batch["orig_size"][0][i].item(), batch["orig_size"][1][i].item())
            pixel_size_mm = batch["pixel_size_mm"][i].item()
            err_mm = pixel_error_to_mm(decoded, gt[i], orig_size, image_size, pixel_size_mm)
            all_errors_mm.append(err_mm)
            all_machines.append(batch["machine"][i])

    return np.stack(all_errors_mm, axis=0), all_machines


def device_breakdown(all_errors_mm, machines, n_bootstrap=1000, min_n_for_ci=10, seed=0):
    """Groups per-sample errors by imaging device and summarizes each group.
    Devices with fewer than min_n_for_ci samples get a flagged warning instead of a
    potentially misleading bootstrap CI — with very few images, a CI is unstable enough
    to be more confusing than informative, and TRIPOD+AI item 12d/23b calls for reporting
    heterogeneity honestly, which includes being upfront about where the data is thin."""
    machines = np.array(machines)
    unique_devices = sorted(set(machines))

    results = []
    for dev in unique_devices:
        mask = machines == dev
        n = mask.sum()
        dev_errors = all_errors_mm[mask]
        flat = dev_errors.flatten()
        mre = flat.mean()
        sdr_2mm = (flat <= 2.0).mean() * 100

        if n >= min_n_for_ci:
            summary = summarize_errors(dev_errors, n_bootstrap=n_bootstrap, seed=seed)
            mre_ci = summary["mre_ci"]
            ci_note = ""
        else:
            mre_ci = None
            ci_note = f" (n={n} — too few images for a reliable CI, interpret with caution)"

        results.append({
            "device": dev, "n": int(n), "mre": mre, "mre_ci": mre_ci,
            "sdr_2mm": sdr_2mm, "ci_note": ci_note,
        })
    return sorted(results, key=lambda d: d["mre"])


def print_device_breakdown(results, title=""):
    print(f"--- {title} ---")
    for r in results:
        ci_str = f"(95% CI: {r['mre_ci'][0]:.3f}-{r['mre_ci'][1]:.3f})" if r["mre_ci"] else ""
        print(f"{r['device']}: n={r['n']}  MRE={r['mre']:.3f}mm {ci_str}{r['ci_note']}  SDR@2mm={r['sdr_2mm']:.1f}%")


def worst_cases(model, dataset, device, image_size=512, heatmap_size=256, n_worst=10):
    """Returns the n_worst samples by per-image mean error, sorted worst-first, along with
    their predicted and ground-truth landmark coordinates — everything needed to plot them
    for a manual clinical review (Phase 7)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    records = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            heatmaps_pred = model(images).cpu()
            gt = batch["landmarks_px_resized"].numpy()

            for i in range(images.size(0)):
                decoded = decode_heatmaps(heatmaps_pred[i], image_size, heatmap_size)
                orig_size = (batch["orig_size"][0][i].item(), batch["orig_size"][1][i].item())
                pixel_size_mm = batch["pixel_size_mm"][i].item()
                err_mm = pixel_error_to_mm(decoded, gt[i], orig_size, image_size, pixel_size_mm)
                records.append({
                    "ceph_id": batch.get("ceph_id", [None] * images.size(0))[i] if "ceph_id" in batch else None,
                    "mean_error_mm": err_mm.mean(),
                    "image": batch["image"][i].numpy(),
                    "pred_landmarks": decoded,
                    "gt_landmarks": gt[i],
                    "machine": batch.get("machine", [None] * images.size(0))[i] if "machine" in batch else None,
                })

    records.sort(key=lambda r: -r["mean_error_mm"])
    return records[:n_worst]


def summarize_errors(all_errors_mm, landmark_names=LANDMARK_ORDER, n_bootstrap=2000, seed=0):
    """all_errors_mm: (N_samples, N_landmarks). Returns MRE, SDR@thresholds, bootstrap CIs,
    and a per-landmark table — everything Phase 4's evaluation harness needs, and everything
    TRIPOD+AI item 23a asks for (performance estimates WITH confidence intervals)."""
    rng = np.random.RandomState(seed)
    n_samples, n_landmarks = all_errors_mm.shape
    flat = all_errors_mm.flatten()  # every (sample, landmark) error, pooled

    mre = flat.mean()
    thresholds = [2.0, 2.5, 3.0, 4.0]
    sdr = {t: (flat <= t).mean() * 100 for t in thresholds}

    # Bootstrap over SAMPLES (not individual landmark errors) — resampling at the image
    # level respects that landmarks within one image aren't independent observations.
    boot_mres = []
    boot_sdrs = {t: [] for t in thresholds}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        resampled = all_errors_mm[idx].flatten()
        boot_mres.append(resampled.mean())
        for t in thresholds:
            boot_sdrs[t].append((resampled <= t).mean() * 100)

    mre_ci = (np.percentile(boot_mres, 2.5), np.percentile(boot_mres, 97.5))
    sdr_ci = {t: (np.percentile(boot_sdrs[t], 2.5), np.percentile(boot_sdrs[t], 97.5)) for t in thresholds}

    per_landmark = []
    for i, name in enumerate(landmark_names):
        lm_errors = all_errors_mm[:, i]
        per_landmark.append({
            "landmark": name,
            "mre": lm_errors.mean(),
            "sdr_2mm": (lm_errors <= 2.0).mean() * 100,
        })

    return {
        "mre": mre, "mre_ci": mre_ci,
        "sdr": sdr, "sdr_ci": sdr_ci,
        "per_landmark": per_landmark,
        "n_samples": n_samples,
    }


def print_summary(summary, title=""):
    print(f"--- {title} ---")
    print(f"N = {summary['n_samples']} samples")
    print(f"MRE: {summary['mre']:.3f} mm  (95% CI: {summary['mre_ci'][0]:.3f}–{summary['mre_ci'][1]:.3f})")
    for t in [2.0, 2.5, 3.0, 4.0]:
        ci = summary['sdr_ci'][t]
        print(f"SDR@{t}mm: {summary['sdr'][t]:.2f}%  (95% CI: {ci[0]:.2f}–{ci[1]:.2f})")
    print()
    print("Worst 5 landmarks by MRE:")
    worst = sorted(summary["per_landmark"], key=lambda d: -d["mre"])[:5]
    for d in worst:
        print(f"  {d['landmark']}: {d['mre']:.3f} mm  (SDR@2mm: {d['sdr_2mm']:.1f}%)")


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"]


def run_training(root, output_dir, n_epochs=100, batch_size=16, lr=1e-3,
                  image_size=512, heatmap_size=256, sigma=4.0, resume=True, model_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "checkpoint.pth")

    train_ds = AarizDataset(root, "train", "Junior Orthodontists", image_size, heatmap_size, sigma)
    val_ds = AarizDataset(root, "valid", "Junior Orthodontists", image_size, heatmap_size, sigma)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    if model_class is None:
        model_class = UNet  # default, keeps existing calls working unchanged
    model = model_class(n_landmarks=29).to(device)
    print(f"Model architecture: {model_class.__name__}, params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedHeatmapMSE(pos_weight=100.0)

    start_epoch = 0
    if resume and os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(model, optimizer, ckpt_path, device) + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    best_val_mre = float("inf")
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        save_checkpoint(model, optimizer, epoch, ckpt_path)  # every epoch — cheap insurance against disconnects

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            val_errors = evaluate(model, val_ds, device, image_size, heatmap_size)
            val_summary = summarize_errors(val_errors, n_bootstrap=200)  # fewer bootstraps during training, just for monitoring
            print(f"Epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.5f}  "
                  f"val_MRE={val_summary['mre']:.3f}mm  val_SDR@2mm={val_summary['sdr'][2.0]:.1f}%  "
                  f"({time.time()-t0:.1f}s)")
            if val_summary["mre"] < best_val_mre:
                best_val_mre = val_summary["mre"]
                save_checkpoint(model, optimizer, epoch, os.path.join(output_dir, "best_model.pth"))
        else:
            print(f"Epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.5f}  ({time.time()-t0:.1f}s)")

    return model


if __name__ == "__main__":
    # Smoke test: run 1 tiny epoch on synthetic data to catch wiring bugs before real training.
    import sys
    sys.path.insert(0, ".")
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, n=8):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {
                "image": torch.rand(3, 512, 512),
                "heatmaps": torch.rand(29, 128, 128),
                "landmarks_px_resized": torch.rand(29, 2) * 512,
                "orig_size": (1200, 960),
                "pixel_size_mm": 0.1,
                "machine": "test",
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_landmarks=29).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = WeightedHeatmapMSE(pos_weight=100.0)

    dummy_train = DummyDataset(n=8)
    loader = DataLoader(dummy_train, batch_size=4)
    loss = train_one_epoch(model, loader, optimizer, criterion, device)
    print("Smoke-test training loss:", loss)

    dummy_val = DummyDataset(n=6)
    errors = evaluate(model, dummy_val, device)
    print("Smoke-test errors shape:", errors.shape)
    assert errors.shape == (6, 29)

    summary = summarize_errors(errors, n_bootstrap=100)
    print_summary(summary, title="Smoke test")

    # Checkpoint save/load round trip
    save_checkpoint(model, optimizer, epoch=0, path="/tmp/smoke_ckpt.pth")
    resumed_epoch = load_checkpoint(model, optimizer, "/tmp/smoke_ckpt.pth", device)
    assert resumed_epoch == 0
    print("Checkpoint save/load round trip OK")
    print()
    print("ALL SMOKE TESTS PASSED")
