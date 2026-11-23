"""
Llama-3.1 RoPE three-segment frequency scaling -- dimension vs frequency plots.

Configuration being visualised:
    head_dim d = 128, rope_theta = 500000, factor alpha = 8,
    low_freq_factor = 1, high_freq_factor = 4,
    original_max_position_embeddings L = 8192

Outputs (labels in English only):
    fig1_rope_dim_vs_f_log.png      f axis on a log scale
    fig2_rope_dim_vs_f_linear.png   f axis on a linear scale (for comparison)

Run:  python3 rope_three_segment_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, MultipleLocator

# ---------------------------------------------------------------- parameters
D = 128              # head_dim
THETA = 500000.0     # rope_theta
FACTOR = 8.0         # rope_scaling["factor"]
LOW = 1.0            # low_freq_factor
HIGH = 4.0           # high_freq_factor
L_ORIG = 8192.0      # original_max_position_embeddings

N = D // 2                              # 64 frequency slots
I = np.arange(N)                        # dimension index
F = THETA ** (-2.0 * I / D)             # original angular frequency (rad / token)
LAM = 2.0 * np.pi / F                   # wavelength (tokens per full turn)

LAM_LOW = L_ORIG / LOW                  # 8192 -> divide-by-alpha branch
LAM_HIGH = L_ORIG / HIGH                # 2048 -> keep-unchanged branch


def llama3_scaled(f, lam, factor=FACTOR, low=LOW, high=HIGH, l_orig=L_ORIG):
    """Llama-3 style three-segment scaling; returns (f', s, k)."""
    f_scaled = np.empty_like(f)
    s_out = np.zeros_like(f)
    k_out = np.ones_like(f)
    lam_low, lam_high = l_orig / low, l_orig / high
    for i, (fi, li) in enumerate(zip(f, lam)):
        if li > lam_low:                      # very low frequency: f / alpha
            fs, s, k = fi / factor, 0.0, factor
        elif li < lam_high:                   # high frequency: untouched
            fs, s, k = fi, 1.0, 1.0
        else:                                 # transition band: convex blend
            s = (l_orig / li - low) / (high - low)
            fs = (1.0 - s) * fi / factor + s * fi
            k = factor / (1.0 - s + factor * s)
        f_scaled[i], s_out[i], k_out[i] = fs, s, k
    return f_scaled, s_out, k_out


F2, S, K = llama3_scaled(F, LAM)
SEG = np.where(LAM > LAM_LOW, 3, np.where(LAM < LAM_HIGH, 1, 2))
n1, n2, n3 = (SEG == 1).sum(), (SEG == 2).sum(), (SEG == 3).sum()

# (y0, y1, label, side)  side keeps the label clear of the curves
BANDS = [
    (-0.5, 29.0, f"segment 1:  f unchanged      ({n1} slots, i = 0..28)", "left"),
    (29.0, 35.0, f"segment 2:  convex blend     ({n2} slots, i = 29..34)", "left"),
    (35.0, 63.5, f"segment 3:  f / alpha = 8    ({n3} slots, i = 35..63)", "right"),
]

C_ORIG = "#1f77b4"
C_SCAL = "#d62728"


def paint_bands(ax, sides=None):
    for idx, (y0, y1, txt, side) in enumerate(BANDS):
        if sides is not None:
            side = sides[idx]
        ax.axhspan(y0, y1, color="0.45", alpha=0.10, zorder=0, lw=0)
        ax.text(0.015 if side == "left" else 0.985, (y0 + y1) / 2.0, txt,
                transform=ax.get_yaxis_transform(),
                ha="left" if side == "left" else "right", va="center",
                fontsize=9, color="0.30", zorder=1)


def base_axis(ax, xlabel):
    ax.set_ylim(63.5, -0.5)                 # i = 0 (high freq) on top
    ax.set_yticks(np.arange(0, 64, 8))
    ax.set_ylabel(r"dimension index $i$   (top = high frequency)", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, which="major", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------------- figure 1: log x axis
fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=150)
paint_bands(ax)

ax.plot(F, I, "o--", color=C_ORIG, lw=2.0, ms=4.5, label=r"original  $f_i$")
ax.plot(F2, I, "o-", color=C_SCAL, lw=2.0, ms=4.5, label=r"scaled  $f_i'$  ($\alpha$ = 8)")

ax.set_xscale("log")
ax.set_xlim(1e-7, 2.0)
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda v, _: f"$10^{{{int(np.log10(v))}}}$" if v > 0 else "")
)
base_axis(ax, r"$f_i$  (rad / token,  log scale)")

ax.annotate("identical for i <= 28\n(segment 1 untouched)",
            xy=(3.30e-3, 28.2), xytext=(1.2e-3, 23.0), fontsize=9, color="0.25",
            ha="right", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="0.45", shrinkB=4))

ax.annotate("seam i = 34 -> 35\nf ratio 1.868  (elsewhere 1.228)",
            xy=(8.5e-4, 34.6), xytext=(1.0e-4, 46.0), fontsize=9, color="0.25",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="0.45", shrinkB=4))

ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95)
ax.set_title(r"RoPE three-segment scaling:  $f_i$ across dimensions  (log $f$ axis)",
             fontsize=13, fontweight="bold", pad=10)
fig.text(0.5, 0.008,
         "alpha=8, low_freq_factor=1, high_freq_factor=4, "
         "original_max_position_embeddings=8192, theta=500000, head_dim=128",
         ha="center", fontsize=8.5, color="0.4")
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig("./images/fig1_rope_dim_vs_f_log.png", bbox_inches="tight")
plt.close(fig)
print("saved fig1_rope_dim_vs_f_log.png")


# --------------------------------------------------- figure 2: linear x axis
fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=150)
paint_bands(ax, sides=("right", "right", "right"))

ax.plot(F, I, "o--", color=C_ORIG, lw=2.0, ms=4.5, label=r"original  $f_i$")
ax.plot(F2, I, "o-", color=C_SCAL, lw=2.0, ms=4.5, label=r"scaled  $f_i'$  ($\alpha$ = 8)")

ax.set_xlim(0, 1.05)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
base_axis(ax, r"$f_i$  (rad / token,  linear scale)")

for i_mark, dx in ((10, 0.06), (20, 0.05), (28, 0.04)):
    ax.annotate(f"i={i_mark}:  $f$={F[i_mark]:.4g}\n= {F[i_mark] / F[0] * 100:.1f}% of the axis",
                xy=(F[i_mark], i_mark), xytext=(F[i_mark] + dx, i_mark + 1.2),
                fontsize=9, color="0.25", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.55", shrinkA=2, shrinkB=4))

ax.annotate("i >= 29 collapse onto x ~ 0\nlargest scaled $f$ in segment 3 = 9.56e-05\n"
            "(both curves overlap below 1e-3)",
            xy=(0.02, 40.0), xytext=(0.34, 43.5), fontsize=9, color="0.25",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="0.45", shrinkB=4))

ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95)
ax.set_title(r"Same data, linear $f$ axis:  only the first ~15 dimensions stay distinguishable",
             fontsize=13, fontweight="bold", pad=10)
fig.text(0.5, 0.008,
         "Identical parameters to the log version -- only the x axis scale changed",
         ha="center", fontsize=8.5, color="0.4")
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig("./images/fig2_rope_dim_vs_f_linear.png", bbox_inches="tight")
plt.close(fig)
print("saved fig2_rope_dim_vs_f_linear.png")

# ------------------------------------------------------------------ console
print("\nsegment counts:  unchanged=%d  blend=%d  divided=%d" % (n1, n2, n3))
print("decades before %.4f -> after %.4f  (+%.3f = log10 alpha)"
      % (np.log10(F[0] / F[-1]), np.log10(F2[0] / F2[-1]),
         np.log10(F2[0] / F2[-1]) - np.log10(F[0] / F[-1])))
print("f'[34]/f'[35] = %.4f    within-band ratio = %.4f"
      % (F2[34] / F2[35], THETA ** (1.0 / N)))
