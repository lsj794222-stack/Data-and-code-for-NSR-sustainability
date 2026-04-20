# ============================================================
# NSR Sustainability Analysis - Part 3
# Sustainability Index + Figure 6 + Sensitivity Analysis + Tables
# ============================================================

# %% Cell 1: Load all previous results
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')

OUT = os.path.expanduser('~/spacenet-data/sea_ice_project/figures')

results = np.load(os.path.join(OUT, 'seg_monthly_sic.npy'))
cbr_segment = np.load(os.path.join(OUT, 'cbr_segment.npy'))
cbr_total = np.load(os.path.join(OUT, 'cbr_total.npy'))
ess = np.load(os.path.join(OUT, 'ess_scores.npy'))

seg_names = ['S1','S2','S3','S4','S5','S6','S7']
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print("All data loaded!")

# %% Cell 2: Calculate Navigational Safety Rating
clim_sic = np.nanmean(results, axis=1) / 100.0  # fraction, shape (7,12)

# NSR_safe: lower SIC = higher safety
nsr_safe = np.zeros((7, 12))
for si in range(7):
    for mi in range(12):
        sic = clim_sic[si, mi]
        if np.isnan(sic):
            sic = 1.0
        # Safety score: 1 = fully safe (no ice), 0 = very dangerous
        nsr_safe[si, mi] = max(0, 1 - sic)

print("Navigational Safety Rating computed")

# %% Cell 3: Normalize CBR to 0-1
cbr_norm = np.zeros((7, 12))
for si in range(7):
    for mi in range(12):
        # Normalize: -50% to +40% -> 0 to 1
        cbr_norm[si, mi] = max(0, min(1, (cbr_segment[si, mi] + 50) / 90))

# %% Cell 4: AHP Sustainability Index - Baseline weights
w1 = 0.40  # Carbon benefit
w2 = 0.35  # Ecological sensitivity (inverted)
w3 = 0.25  # Navigational safety

SI = np.zeros((7, 12))
for si in range(7):
    for mi in range(12):
        SI[si, mi] = w1 * cbr_norm[si, mi] + w2 * (1 - ess[si]) + w3 * nsr_safe[si, mi]

print("\n=== Table 5: Sustainability Index (Baseline) ===")
print(f"{'Seg':<6}", end='')
for ml in month_labels:
    print(f"{ml:>6}", end='')
print()
print("-" * 80)
for si in range(7):
    print(f"{seg_names[si]:<6}", end='')
    for mi in range(12):
        val = SI[si, mi]
        print(f"{val:>6.2f}", end='')
    print()

# %% Cell 5: Figure 6 - Sustainability Index Heatmap
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

# Custom colormap: red -> yellow -> light green -> dark green
cmap = mcolors.LinearSegmentedColormap.from_list('si_cmap', 
    [(0.0, '#d73027'), (0.3, '#fc8d59'), (0.5, '#fee08b'), 
     (0.6, '#d9ef8b'), (0.7, '#91cf60'), (1.0, '#1a9850')], N=256)

im = ax.imshow(SI, aspect='auto', cmap=cmap, vmin=0, vmax=1)
ax.set_yticks(range(7))
ax.set_yticklabels([f"{s}\n({segments[s]})" if si < 3 else s 
                    for si, s in enumerate(seg_names)] if False else seg_names)
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels, rotation=45)
ax.set_title('Figure 6: Sustainability Index by Segment and Month (2021–2025 Average)', fontsize=13)
plt.colorbar(im, ax=ax, label='Sustainability Index (0=Unsustainable, 1=Optimal)')

# Add text and threshold line
for i in range(7):
    for j in range(12):
        val = SI[i, j]
        color = 'white' if val < 0.3 or val > 0.75 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, 
                color=color, fontweight='bold')

# Draw SI=0.6 contour
from matplotlib.patches import Rectangle
for i in range(7):
    for j in range(12):
        if SI[i, j] >= 0.6:
            rect = Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, 
                            edgecolor='white', facecolor='none', linestyle='--')
            ax.add_patch(rect)

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('NSR Segment', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Fig6_sustainability_index.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'Fig6_sustainability_index.tiff'), dpi=300, bbox_inches='tight')
print("Figure 6 saved!")
plt.show()

# %% Cell 6: Sensitivity Analysis (Figure S1)
weight_scenarios = {
    'Equal': (0.33, 0.33, 0.34),
    'Carbon-priority': (0.60, 0.25, 0.15),
    'Ecology-priority': (0.25, 0.55, 0.20),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (scenario, (ww1, ww2, ww3)) in enumerate(weight_scenarios.items()):
    SI_alt = np.zeros((7, 12))
    for si in range(7):
        for mi in range(12):
            SI_alt[si, mi] = ww1 * cbr_norm[si, mi] + ww2 * (1 - ess[si]) + ww3 * nsr_safe[si, mi]
    
    im = axes[idx].imshow(SI_alt, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    axes[idx].set_yticks(range(7))
    axes[idx].set_yticklabels(seg_names)
    axes[idx].set_xticks(range(12))
    axes[idx].set_xticklabels(month_labels, rotation=45)
    axes[idx].set_title(f'({chr(97+idx)}) {scenario}\nw=({ww1},{ww2},{ww3})', fontsize=11)
    
    # Count sustainable cells
    n_sustainable = np.sum(SI_alt >= 0.6)
    axes[idx].text(0.5, -0.15, f'SI≥0.6: {n_sustainable}/84 cells', 
                  transform=axes[idx].transAxes, ha='center', fontsize=10)
    
    for i in range(7):
        for j in range(12):
            val = SI_alt[i, j]
            color = 'white' if val < 0.3 or val > 0.75 else 'black'
            axes[idx].text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=5, color=color)

plt.colorbar(im, ax=axes, label='SI', shrink=0.8)
plt.suptitle('Figure S1: Sensitivity Analysis of SI Under Alternative Weighting Scenarios', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FigS1_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'FigS1_sensitivity_analysis.tiff'), dpi=300, bbox_inches='tight')
print("Figure S1 saved!")
plt.show()

# %% Cell 7: Figure S2 - Annual SIC time series by segment
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

year_list = list(range(2012, 2026))
for si in range(7):
    ax = axes[si]
    for mi in [6, 7, 8, 9]:  # Jul, Aug, Sep, Oct
        monthly_vals = results[si, :, mi]
        ax.plot(year_list, monthly_vals, '-o', markersize=3, label=month_labels[mi])
    ax.set_title(f'{seg_names[si]}', fontsize=11)
    ax.set_xlabel('Year')
    ax.set_ylabel('SIC (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[7].axis('off')  # empty subplot
plt.suptitle('Figure S2: Monthly SIC Time Series by Segment (Jul–Oct)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FigS2_SIC_timeseries.png'), dpi=300, bbox_inches='tight')
print("Figure S2 saved!")
plt.show()

# %% Cell 8: Generate all supplementary tables as CSV
# Table S1: Complete monthly SIC
rows = []
for si in range(7):
    for yi, year in enumerate(range(2012, 2026)):
        row = {'Segment': seg_names[si], 'Year': year}
        for mi in range(12):
            row[month_labels[mi]] = results[si, yi, mi]
        rows.append(row)
df_s1 = pd.DataFrame(rows)
df_s1.to_csv(os.path.join(OUT, 'TableS1_monthly_SIC.csv'), index=False)
print("Table S1 saved")

# Table S4: Complete SI values
rows = []
for si in range(7):
    row = {'Segment': seg_names[si]}
    for mi in range(12):
        row[month_labels[mi]] = round(SI[si, mi], 3)
    rows.append(row)
df_s4 = pd.DataFrame(rows)
df_s4.to_csv(os.path.join(OUT, 'TableS4_SI_values.csv'), index=False)
print("Table S4 saved")

# Summary statistics for paper
print("\n" + "="*60)
print("KEY RESULTS SUMMARY FOR PAPER")
print("="*60)
print(f"\nSCR CO₂: {6020:.0f} tonnes (reference)")
print(f"NSR CO₂ (August): {np.sum(np.load(os.path.join(OUT,'nsr_co2_monthly.npy'))[:,7]):.0f} + approach/exit")
print(f"NSR CO₂ (September): {np.sum(np.load(os.path.join(OUT,'nsr_co2_monthly.npy'))[:,8]):.0f} + approach/exit")
print(f"\nCBR August: {cbr_total[7]:+.1f}%")
print(f"CBR September: {cbr_total[8]:+.1f}%")
print(f"CBR July: {cbr_total[6]:+.1f}%")
print(f"CBR October: {cbr_total[9]:+.1f}%")
print(f"\nSustainable cells (SI≥0.6): {np.sum(SI >= 0.6)}/84")
print(f"Optimal cells (SI≥0.7): {np.sum(SI >= 0.7)}/84")
print(f"\nBest segment-month: S7-Sep = {SI[6,8]:.2f}")
print(f"Worst segment-month: S3-Jan = {SI[2,0]:.2f}")

# List all output files
print(f"\n=== Output files in {OUT} ===")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f}: {size/1024:.1f} KB")

print("\n" + "="*60)
print("ALL ANALYSIS COMPLETE!")
print("="*60)

# Segment definitions for reference
segments = {
    'S1': 'Bering Strait - Pevek',
    'S2': 'Pevek - New Siberian Is.',
    'S3': 'New Siberian Is. - Vilkitsky',
    'S4': 'Vilkitsky - Dikson',
    'S5': 'Dikson - Novaya Zemlya',
    'S6': 'Novaya Zemlya - Murmansk',
    'S7': 'Murmansk - Norwegian Sea',
}
