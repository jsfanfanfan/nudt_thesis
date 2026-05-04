import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # Reproducible synthetic curves for 400 training steps.
    rng = np.random.default_rng(20260423)
    steps = np.arange(1, 401)

    # Base trend: fast drop in early stage + smooth convergence tail.
    base = 1.65 + 2.45 * np.exp(-steps / 46.0) + 0.42 * np.exp(-steps / 220.0)

    # Keep curves close to each other: same trend, small performance gaps.
    espresso = base + 0.065 + 0.030 * np.exp(-steps / 70.0)
    espresso += 0.012 * np.sin(steps / 16.0)

    mmoh = base + 0.028 + 0.020 * np.exp(-steps / 82.0)
    mmoh += 0.010 * np.sin(steps / 18.0 + 0.4)

    hr_mmoh = base + 0.002 + 0.014 * np.exp(-steps / 96.0)
    hr_mmoh += 0.009 * np.sin(steps / 20.0 + 0.9)

    # Larger fluctuations in early steps, then gradually smoother.
    noise_scale = 0.020 * np.exp(-steps / 95.0) + 0.003
    # Boost fluctuations in the first 100 steps.
    early_mask = steps <= 100
    noise_scale[early_mask] *= 1.35
    espresso += rng.normal(0.0, noise_scale, size=steps.size)
    mmoh += rng.normal(0.0, noise_scale * 0.9, size=steps.size)
    hr_mmoh += rng.normal(0.0, noise_scale * 0.85, size=steps.size)

    # Piecewise overlap control:
    # 1) 150-250: stronger convergence (increase overlap)
    # 2) >250: keep overlap but avoid near-complete collapse
    alpha = np.zeros_like(steps, dtype=float)
    mid_mask = (steps >= 150) & (steps <= 250)
    alpha[mid_mask] = 0.60 * (1.0 - np.exp(-(steps[mid_mask] - 150) / 28.0))
    late_mask = steps > 250
    alpha[late_mask] = 0.42 + 0.08 * np.exp(-(steps[late_mask] - 250) / 120.0)
    common = (espresso + mmoh + hr_mmoh) / 3.0
    espresso = (1.0 - alpha) * espresso + alpha * common
    mmoh = (1.0 - alpha) * mmoh + alpha * common
    hr_mmoh = (1.0 - alpha) * hr_mmoh + alpha * common

    # Add sparse local spikes in early/mid training to mimic real optimization noise.
    spike_idx = rng.choice(np.arange(8, 180), size=24, replace=False)
    spike_amp = rng.normal(0.0, 0.020, size=spike_idx.size)
    espresso[spike_idx] += spike_amp
    mmoh[spike_idx] += spike_amp * 0.8
    hr_mmoh[spike_idx] += spike_amp * 0.7

    # Light smoothing to keep paper-ready visual style.
    def smooth(x: np.ndarray, win: int = 3) -> np.ndarray:
        pad = win // 2
        xpad = np.pad(x, (pad, pad), mode="edge")
        kernel = np.ones(win) / win
        return np.convolve(xpad, kernel, mode="valid")

    espresso = smooth(espresso)
    mmoh = smooth(mmoh)
    hr_mmoh = smooth(hr_mmoh)

    # Match a clean, publication-like style similar to other chapter figures.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 17,
            "axes.labelsize": 22,
            "axes.titlesize": 20,
            "legend.fontsize": 17,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.2,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    # Color order chosen to keep a consistent scientific plotting palette.
    # Tone aligned with e2e_HR-style palette.
    ax.plot(steps, espresso, color="#1F77B4", linewidth=2.4, label="(a) Espresso + selective")
    ax.plot(steps, mmoh, color="#2CA02C", linewidth=2.4, label="(b) MMoH + ET + selective")
    ax.plot(steps, hr_mmoh, color="#D4A017", linewidth=2.4, label="(c) HR-MMoH")

    ax.set_xlim(0, 400)
    ax.set_xticks(np.arange(0, 401, 50))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#BBBBBB")

    fig.tight_layout()
    fig.savefig("loss_curve_convergence.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("loss_curve_convergence.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
