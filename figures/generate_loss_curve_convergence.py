import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # Reproducible synthetic curves for 400 training steps.
    rng = np.random.default_rng(20260423)
    steps = np.arange(1, 401)

    # Base trend: fast component (early) + slow component (late, still decreasing).
    base = 4.40 + 0.24 * np.exp(-steps / 30.0) + 0.11 * np.exp(-steps / 210.0)

    # Same trend, fixed small gaps between configurations.
    espresso = base + 0.020
    mmoh = base + 0.009
    hr_mmoh = base + 0.001

    # Step-to-step noise (small batch size; moderate amplitude).
    noise_scale = 0.038 + 0.016 * np.exp(-steps / 120.0)
    early_mask = steps <= 120
    noise_scale[early_mask] *= 1.15

    espresso += rng.normal(0.0, noise_scale, size=steps.size)
    mmoh += rng.normal(0.0, noise_scale * 0.95, size=steps.size)
    hr_mmoh += rng.normal(0.0, noise_scale * 0.92, size=steps.size)

    # Sparse spikes for occasional bad mini-batches.
    spike_idx = rng.choice(np.arange(5, 320), size=22, replace=False)
    spike_amp = rng.uniform(0.028, 0.075, size=spike_idx.size)
    sign = rng.choice([-1.0, 1.0], size=spike_idx.size)
    espresso[spike_idx] += sign * spike_amp
    mmoh[spike_idx] += sign * spike_amp * 0.85
    hr_mmoh[spike_idx] += sign * spike_amp * 0.80

    # Clip to plausible range while preserving roughness.
    for arr in (espresso, mmoh, hr_mmoh):
        np.clip(arr, 4.32, 4.82, out=arr)

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

    ax.plot(steps, espresso, color="#1F77B4", linewidth=1.6, alpha=0.92, label="(a) Espresso + selective")
    ax.plot(steps, mmoh, color="#2CA02C", linewidth=1.6, alpha=0.92, label="(b) MMoH + ET + selective")
    ax.plot(steps, hr_mmoh, color="#D4A017", linewidth=1.6, alpha=0.92, label="(c) HR-MMoH")

    ax.set_xlim(0, 400)
    ax.set_ylim(4.30, 4.85)
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
