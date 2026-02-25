"""
Benchmark: MAE2D vs MAE3D Computational Cost on OpenCell

Measures:
  - Parameter count
  - Token counts (encoder & decoder)
  - Patch embedding dimensions
  - Per-step training time (forward + backward + optimizer)
  - Inference time (forward only)
  - Peak GPU memory (training & inference)

Run from repo root:
    python src/benchmark_mae2d_vs_mae3d.py
    python src/benchmark_mae2d_vs_mae3d.py --device cpu   # CPU-only fallback
"""

import sys
import os
import argparse
import time
import gc

import torch
import torch.nn as nn
import numpy as np

# ── resolve imports ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from lib.models.mae2d import MAE2D
from lib.models.mae3d import MAE3D
from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
from lib.networks.patch_embed_layers import PatchEmbed2D, PatchEmbed3D

# ── config namespace ─────────────────────────────────────────────────────────
class Cfg:
    """Simple namespace matching the OmegaConf keys used by MAE2D / MAE3D."""
    pass


def make_2d_args():
    a = Cfg()
    a.input_size   = [176, 176]
    a.patch_size   = [8, 8]
    a.in_chans     = 2
    a.mask_ratio   = 0.75
    a.pos_embed_type = "sincos"
    a.encoder_embed_dim = 384
    a.encoder_depth     = 6
    a.encoder_num_heads = 6
    a.decoder_embed_dim = 192
    a.decoder_depth     = 4
    a.decoder_num_heads = 6
    a.patchembed = "PatchEmbed2D"
    return a


def make_3d_args():
    a = Cfg()
    a.input_size   = [100, 176, 176]
    a.patch_size   = [10, 8, 8]
    a.in_chans     = 2
    a.mask_ratio   = 0.75
    a.pos_embed_type = "sincos"
    a.encoder_embed_dim = 384
    a.encoder_depth     = 6
    a.encoder_num_heads = 6
    a.decoder_embed_dim = 192
    a.decoder_depth     = 4
    a.decoder_num_heads = 6
    a.patchembed = "PatchEmbed3D"
    return a


# ── helpers ──────────────────────────────────────────────────────────────────
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return float("nan")


def warmup_and_time(fn, n_warmup=3, n_runs=10):
    """Run fn() n_warmup times for warmup, then n_runs times and return
    mean and std wall-clock time in milliseconds."""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - start) * 1e3)

    return float(np.mean(times)), float(np.std(times))


def print_section(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_row(label, val2d, val3d, fmt="{}", unit=""):
    label_w = 38
    col_w   = 14
    v2 = fmt.format(val2d) + unit
    v3 = fmt.format(val3d) + unit
    ratio = ""
    if isinstance(val2d, (int, float)) and isinstance(val3d, (int, float)) and val2d != 0:
        ratio = f"  ({val3d/val2d:.1f}x)"
    print(f"  {label:<{label_w}}{v2:>{col_w}}{v3:>{col_w}}{ratio}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch2d", type=int, default=8,  help="batch size for MAE2D (default: 8, matches config)")
    parser.add_argument("--batch3d", type=int, default=2,  help="batch size for MAE3D (default: 2, matches config)")
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_runs",   type=int, default=10)
    args_cli = parser.parse_args()

    device = torch.device(args_cli.device)
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(device)}")

    # ── build models ─────────────────────────────────────────────────────────
    args2d = make_2d_args()
    args3d = make_3d_args()

    # Monkey-patch lib.networks.patch_embed_layers so getattr lookup works
    import lib.networks.patch_embed_layers as pel
    pel.PatchEmbed2D = PatchEmbed2D
    pel.PatchEmbed3D = PatchEmbed3D

    model2d = MAE2D(encoder=MAEViTEncoder, decoder=MAEViTDecoder, args=args2d).to(device)
    model3d = MAE3D(encoder=MAEViTEncoder, decoder=MAEViTDecoder, args=args3d).to(device)
    model2d.eval()
    model3d.eval()

    # ── structural stats ─────────────────────────────────────────────────────
    g2 = model2d.grid_size            # [22, 22]
    g3 = model3d.grid_size            # [10, 22, 22]
    n2 = int(np.prod(g2))             # 484
    n3 = int(np.prod(g3))             # 4840
    sel2 = int(n2 * (1 - args2d.mask_ratio))   # 121 visible
    sel3 = int(n3 * (1 - args3d.mask_ratio))   # 1210 visible
    patch_dim2 = int(args2d.in_chans * np.prod(args2d.patch_size))   # 2*8*8 = 128
    patch_dim3 = int(args3d.in_chans * np.prod(args3d.patch_size))   # 2*10*8*8 = 1280

    total2, train2 = count_params(model2d)
    total3, train3 = count_params(model3d)

    print_section("Architecture & Token Counts")
    header_label = ""
    print(f"  {'Metric':<38}{'MAE2D':>14}{'MAE3D':>14}  Ratio")
    print(f"  {'-'*66}")
    print_row("Input shape",
              f"(B,2,176,176)", f"(B,2,100,176,176)", fmt="{}")
    print_row("Patch size",
              str(args2d.patch_size), str(args3d.patch_size), fmt="{}")
    print_row("Grid size",
              str(g2), str(g3), fmt="{}")
    print_row("Total patches (N)",      n2, n3, fmt="{:,}")
    print_row("Visible tokens (encoder, 25%)", sel2+1, sel3+1, fmt="{:,}",
              unit=" (+CLS)")
    print_row("Decoder tokens (all patches)", n2, n3, fmt="{:,}")
    print_row("Patch embedding dim",    patch_dim2, patch_dim3, fmt="{:,}")
    print_row("Encoder embed dim",
              args2d.encoder_embed_dim, args3d.encoder_embed_dim, fmt="{:,}")
    print_row("Encoder depth / heads",
              f"{args2d.encoder_depth}/{args2d.encoder_num_heads}",
              f"{args3d.encoder_depth}/{args3d.encoder_num_heads}", fmt="{}")
    print_row("Decoder embed dim",
              args2d.decoder_embed_dim, args3d.decoder_embed_dim, fmt="{:,}")
    print_row("Decoder depth / heads",
              f"{args2d.decoder_depth}/{args2d.decoder_num_heads}",
              f"{args3d.decoder_depth}/{args3d.decoder_num_heads}", fmt="{}")
    print_row("Total parameters",       total2/1e6, total3/1e6, fmt="{:.2f}", unit="M")
    print_row("Trainable parameters",   train2/1e6, train3/1e6, fmt="{:.2f}", unit="M")

    # ── theoretical attention cost ────────────────────────────────────────────
    # Attention cost ∝ N^2 * d for each layer
    enc_attn2 = (sel2 + 1)**2 * args2d.encoder_embed_dim * args2d.encoder_depth
    enc_attn3 = (sel3 + 1)**2 * args3d.encoder_embed_dim * args3d.encoder_depth
    dec_attn2 = n2**2 * args2d.decoder_embed_dim * args2d.decoder_depth
    dec_attn3 = n3**2 * args3d.decoder_embed_dim * args3d.decoder_depth

    print_section("Theoretical Attention Cost (∝ N² · d · layers)")
    print(f"  {'Metric':<38}{'MAE2D':>14}{'MAE3D':>14}  Ratio")
    print(f"  {'-'*66}")
    print_row("Encoder attention ops (×10⁹)", enc_attn2/1e9, enc_attn3/1e9, fmt="{:.3f}")
    print_row("Decoder attention ops (×10⁹)", dec_attn2/1e9, dec_attn3/1e9, fmt="{:.3f}")
    print_row("Total attention ops (×10⁹)",
              (enc_attn2+dec_attn2)/1e9,
              (enc_attn3+dec_attn3)/1e9, fmt="{:.3f}")

    # ── timing: inference ─────────────────────────────────────────────────────
    print_section(f"Inference Time  (batch2d={args_cli.batch2d}, batch3d={args_cli.batch3d})")

    x2 = torch.randn(args_cli.batch2d, 2, 176, 176,       device=device)
    x3 = torch.randn(args_cli.batch3d, 2, 100, 176, 176,  device=device)

    model2d.eval(); model3d.eval()

    def inf2d():
        with torch.no_grad():
            model2d(x2)
    def inf3d():
        with torch.no_grad():
            model3d(x3)

    reset_peak_memory(device)
    inf2d()
    mem_inf2 = peak_memory_mb(device)

    reset_peak_memory(device)
    inf3d()
    mem_inf3 = peak_memory_mb(device)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t2_mean, t2_std = warmup_and_time(inf2d, args_cli.n_warmup, args_cli.n_runs)
    t3_mean, t3_std = warmup_and_time(inf3d, args_cli.n_warmup, args_cli.n_runs)

    print(f"  {'Metric':<38}{'MAE2D':>14}{'MAE3D':>14}  Ratio")
    print(f"  {'-'*66}")
    print_row("Fwd time / step (ms)",  t2_mean, t3_mean, fmt="{:.1f}")
    print(f"  {'  ± std dev (ms)':<38}{'± '+f'{t2_std:.1f}':>14}{'± '+f'{t3_std:.1f}':>14}")
    print_row("Fwd time / sample (ms)",
              t2_mean/args_cli.batch2d, t3_mean/args_cli.batch3d, fmt="{:.1f}")
    if device.type == "cuda":
        print_row("Peak GPU mem (MB)", mem_inf2, mem_inf3, fmt="{:.0f}")

    # ── timing: training step ─────────────────────────────────────────────────
    print_section(f"Training Time  (batch2d={args_cli.batch2d}, batch3d={args_cli.batch3d})")

    model2d.train(); model3d.train()
    opt2 = torch.optim.AdamW(model2d.parameters(), lr=1.5e-4)
    opt3 = torch.optim.AdamW(model3d.parameters(), lr=1.5e-4)

    def train2d():
        opt2.zero_grad(set_to_none=True)
        loss = model2d(x2)
        loss.backward()
        opt2.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    def train3d():
        opt3.zero_grad(set_to_none=True)
        loss = model3d(x3)
        loss.backward()
        opt3.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    reset_peak_memory(device)
    train2d()
    mem_train2 = peak_memory_mb(device)

    reset_peak_memory(device)
    train3d()
    mem_train3 = peak_memory_mb(device)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tt2_mean, tt2_std = warmup_and_time(train2d, args_cli.n_warmup, args_cli.n_runs)
    tt3_mean, tt3_std = warmup_and_time(train3d, args_cli.n_warmup, args_cli.n_runs)

    print(f"  {'Metric':<38}{'MAE2D':>14}{'MAE3D':>14}  Ratio")
    print(f"  {'-'*66}")
    print_row("Fwd+Bwd time / step (ms)", tt2_mean, tt3_mean, fmt="{:.1f}")
    print(f"  {'  ± std dev (ms)':<38}{'± '+f'{tt2_std:.1f}':>14}{'± '+f'{tt3_std:.1f}':>14}")
    print_row("Fwd+Bwd time / sample (ms)",
              tt2_mean/args_cli.batch2d, tt3_mean/args_cli.batch3d, fmt="{:.1f}")
    if device.type == "cuda":
        print_row("Peak GPU mem training (MB)", mem_train2, mem_train3, fmt="{:.0f}")

    # ── data volume ───────────────────────────────────────────────────────────
    bytes2 = args_cli.batch2d * 2 * 176 * 176 * 4          # float32
    bytes3 = args_cli.batch3d * 2 * 100 * 176 * 176 * 4
    print_section("Data Volume per Step")
    print(f"  {'Metric':<38}{'MAE2D':>14}{'MAE3D':>14}  Ratio")
    print(f"  {'-'*66}")
    print_row("Input tensor (MB)", bytes2/1e6, bytes3/1e6, fmt="{:.1f}")

    print("\n")


if __name__ == "__main__":
    main()
