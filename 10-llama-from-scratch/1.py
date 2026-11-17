import math, torch
import matplotlib.pyplot as plt

def rope_f(d=128, theta=500000., alpha=8., lo=1., hi=4., L=8192.):
    i = torch.arange(0, d, 2, dtype=torch.float64)
    f0 = theta ** (-i / d)
    lam = 2 * math.pi / f0
    s = ((L / lam - lo) / (hi - lo)).clamp(0., 1.)
    f1 = torch.where(lam > L / lo, f0 / alpha,
         torch.where(lam < L / hi, f0,
                     (1 - s) * (f0 / alpha) + s * f0))
    return f0, f1

f0, f1 = rope_f()
x = torch.arange(0, 128, 2)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.semilogy(x, f0, '--', lw=1.9, label='original f_i')
ax.semilogy(x, f1, '-',  lw=1.9, label="scaled f_i'")
ax.axvspan(-0.5, 28.5, alpha=.06)
ax.axvspan(28.5, 34.5, alpha=.14)
ax.axvspan(34.5, 63.5, alpha=.06)
for tx, name in [(14, '1 unchanged'), (31.5, '2 blend'), (49, '3 / 8')]:
    ax.text(tx, f0.max() * 1.3, name, ha='center', fontsize=9)
ax.set_xlabel('dimension index i')
ax.set_ylabel('f_i  (rad / token)')
ax.set_title('RoPE three-segment f_i: theta=500000, d=128, alpha=8')
ax.legend(fontsize=9)
ax.set_ylim(float(f1.min()) / 3, float(f0.max()) * 3)
plt.tight_layout()
plt.savefig('rope_fi.png', dpi=160)
