"""Generate the conceptual figures used by README (pure drawing, no training).

    python assets/make_figures.py

Outputs (created in ./assets, the folder is made if missing):
    assets/grpo_pipeline.png    one GRPO iteration, end to end
    assets/group_advantage.png  group Z-score: normal group vs dead group
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# assets/ next to (or one level above) this script, whichever exists after creation
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE if os.path.basename(HERE) == "assets" else os.path.join(os.path.dirname(HERE), "assets")

BLUE, ORANGE, GREEN, GRAY = "#2a6fdb", "#d95f02", "#1b9e77", "#8da0cb"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("written:", path)


def box(ax, x, y, w, h, text, edge=BLUE, fc="#f3f7ff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.4, edgecolor=edge, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, linespacing=1.5)


def arrow(ax, xy1, xy2, color=GRAY, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=14,
                                 linewidth=1.4, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0.6, 4.15)
    ax.axis("off")

    w, h, y = 2.0, 1.0, 2.55
    xs = [0.35, 2.75, 5.15, 7.55, 9.95]
    labels = [
        "prompt\n\"3+9=\"",
        "policy $\\pi_\\theta$\nsample $G$ responses",
        "rule verifier\n$r_i \\in \\{0,\\,0.2,\\,1\\}$",
        "group normalize\n$A_i=(r_i-\\bar r)/\\sigma_r$",
        "token-level clip\n+ KL(k3), update $\\theta$",
    ]
    edges = [BLUE, BLUE, ORANGE, GREEN, GREEN]
    for x, t, c in zip(xs, labels, edges):
        box(ax, x, y, w, h, t, edge=c, fc="#f7f9ff")

    for i in range(4):
        arrow(ax, (xs[i] + w + 0.03, y + h / 2), (xs[i + 1] - 0.03, y + h / 2))

    # feedback: resample with the updated parameters (on-policy)
    arrow(ax, (xs[4] + w / 2, y - 0.05), (xs[1] + w / 2, y - 0.05), color=GRAY,
          style="-|>", rad=-0.22, ls=(0, (4, 2)))
    ax.text(6.1, 1.35, "resample with the updated $\\theta$  (on-policy: a batch is reused for a few epochs only)",
            ha="center", fontsize=9.5, color="#555555")

    ax.text(0.35, 3.92, "One GRPO iteration: no critic, no reward model", fontsize=12.5, weight="bold")
    ax.text(0.35, 0.82, "The $G$ responses to one prompt share a single scalar reward "
                        "→ group-relative advantage → broadcast to every token of that response",
            fontsize=9.5, color="#555555")

    fig.tight_layout()
    save(fig, "grpo_pipeline.png")


def group_advantage():
    import numpy as np

    r_ok = np.array([1.0, 0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2])
    r_dead = np.array([1.0] * 8)
    mu_ok, sd_ok = r_ok.mean(), r_ok.std()
    a_ok = (r_ok - mu_ok) / sd_ok
    a_dead = np.zeros(8)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    i = np.arange(8)

    for col, (title, r, a) in enumerate([
        ("Normal group: rewards differ", r_ok, a_ok),
        ("Dead group: all 8 correct, identical rewards", r_dead, a_dead),
    ]):
        ax = axes[0, col]
        ax.bar(i, r, color=ORANGE, alpha=0.85, width=0.62)
        ax.axhline(r.mean(), color=GRAY, linewidth=1.3, linestyle="--")
        ax.text(7.4, r.mean() + 0.03, f"group mean {r.mean():.3f}", ha="right", fontsize=9, color="#555555")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("reward $r_i$")
        ax.set_ylim(-0.05, 1.25)
        ax.grid(alpha=0.2, axis="y")

        ax = axes[1, col]
        colors = [GREEN if v > 0 else BLUE for v in a]
        ax.bar(i, a, color=colors, alpha=0.85, width=0.62)
        ax.axhline(0, color="#333333", linewidth=0.9)
        for k, v in enumerate(a):
            if col == 0:
                ax.text(k, v + (0.12 if v >= 0 else -0.28), f"{v:+.2f}", ha="center", fontsize=8)
        ax.set_ylabel("advantage $A_i$")
        ax.set_xlabel("response $i$")
        ax.set_ylim(-0.9, 2.6)
        ax.grid(alpha=0.2, axis="y")

    axes[0, 0].text(0.02, 0.96, "$A=(r-\\mathrm{mean})/\\mathrm{std}$", transform=axes[0, 0].transAxes,
                    fontsize=9, color="#555555", va="top")
    axes[0, 1].text(0.02, 0.96, "$\\mathrm{std}=0$ → all $A=0$\nno signal, this group is wasted",
                    transform=axes[0, 1].transAxes, fontsize=9, color="#555555", va="top")
    axes[1, 1].text(0.5, 0.5, "all advantages are 0: zero gradient from this group", ha="center", va="center",
                    transform=axes[1, 1].transAxes, fontsize=10, color="#555555")

    fig.suptitle("Group Z-score: relative scores within a prompt replace the critic's absolute value estimates",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "group_advantage.png")


if __name__ == "__main__":
    pipeline()
    group_advantage()
