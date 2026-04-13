import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Style：与主图一致 ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})

BASE   = "/home1/zhln/code/BadDiffusion"
path_b = f"{BASE}/res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT/unet/diffusion_pytorch_model.bin"
path_c = f"{BASE}/res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT/unet/diffusion_pytorch_model.bin"

sd_b = torch.load(path_b, map_location='cpu')
sd_c = torch.load(path_c, map_location='cpu')

tau_flat = []
for key in sd_b:
    if key in sd_c:
        diff = (sd_b[key].float() - sd_c[key].float()).abs().flatten()
        tau_flat.append(diff)
tau = torch.cat(tau_flat).numpy()

p99  = np.percentile(tau, 99)
p999 = np.percentile(tau, 99.9)

# ── 能量数据 ─────────────────────────────────────────────────────────────────
ks     = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
labels = ['0.001%','0.005%','0.01%','0.05%','0.1%',
          '0.2%','0.5%','1%','5%','10%','20%','50%']
energies = []
for k in ks:
    thresh = np.percentile(tau, 100 - k)
    energies.append(tau[tau >= thresh].sum() / tau.sum() * 100)

# ── 画图 ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

# 左：分布直方图
ax1.hist(tau, bins=500, log=True, color='#1f77b4', alpha=0.75, edgecolor='none')
ax1.axvline(p99,  color='#d62728', ls='--', lw=1.5,
            label=f'Top 1%   ($|\\tau|\\geq{p99:.4f}$)')
ax1.axvline(p999, color='#ff7f0e', ls='--', lw=1.5,
            label=f'Top 0.1% ($|\\tau|\\geq{p999:.4f}$)')
ax1.set_xlabel(r'$|\tau_i|$', labelpad=4)
ax1.set_ylabel('Count (log scale)')
ax1.set_title('(a) Distribution of Task Vector Magnitude', pad=8)
ax1.legend(framealpha=0.85)

# 右：能量集中曲线
x_pos = list(range(len(ks)))
ax2.plot(x_pos, energies, marker='o', color='#1f77b4', lw=1.8, ms=6)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel(r'% of total $|\tau|$ energy')
ax2.set_title('(b) Task Vector Energy Concentration', pad=8)
ax2.set_ylim(0, 90)
# 只加轻微的水平参考线
for y in [20, 40, 60, 80]:
    ax2.axhline(y, color='grey', lw=0.5, alpha=0.4, ls='--')

fig.suptitle('Task Vector Analysis (CIFAR-10)', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('tau_analysis_cifar10.pdf', bbox_inches='tight')
fig.savefig('tau_analysis_cifar10.png', bbox_inches='tight', dpi=300)
print("Saved: tau_analysis_cifar10.pdf / .png")