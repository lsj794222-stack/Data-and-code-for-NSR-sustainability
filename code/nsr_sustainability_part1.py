# ============================================================
# NSR Sustainability Analysis - Part 1
# Environment Setup + Figure 1 (Study Area) + Figure 3 (SIC Analysis)
# ============================================================

# %% Cell 1: Install packages
import subprocess, sys
for pkg in ['xarray','netCDF4','scipy','matplotlib','pandas','numpy','h5py','hdf5storage','cmocean']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

# %% Cell 2: Imports and paths
import os, glob, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import loadmat
import hdf5storage
warnings.filterwarnings('ignore')

BASE = os.path.expanduser('~/spacenet-data/sea_ice_project/data')
AMSR2 = os.path.join(BASE, 'amsr2')
ERA5 = os.path.join(BASE, 'era5')
ADD = os.path.join(BASE, 'additional')
OUT = os.path.expanduser('~/spacenet-data/sea_ice_project/figures')
os.makedirs(OUT, exist_ok=True)

print("Paths OK")
print(f"AMSR2 years: {sorted(os.listdir(AMSR2))[:5]}...")

# %% Cell 3: Load grid coordinates from grid_lonlat.mat
grid_file = os.path.join(AMSR2, 'grid_lonlat.mat')
try:
    grid = hdf5storage.loadmat(grid_file)
    print("Loaded with hdf5storage. Keys:", list(grid.keys()))
except:
    grid = loadmat(grid_file)
    print("Loaded with scipy. Keys:", list(grid.keys()))

# Extract lon/lat - try common variable names
for k in grid.keys():
    if 'lon' in k.lower():
        lon_grid = grid[k]
        print(f"Longitude: {k}, shape={lon_grid.shape}")
    if 'lat' in k.lower():
        lat_grid = grid[k]
        print(f"Latitude: {k}, shape={lat_grid.shape}")

# %% Cell 4: Load NSR route
route_file = os.path.join(AMSR2, 'nsr_route.mat')
try:
    route = hdf5storage.loadmat(route_file)
except:
    route = loadmat(route_file)
print("NSR route keys:", list(route.keys()))
for k in route.keys():
    if not k.startswith('_'):
        print(f"  {k}: shape={np.array(route[k]).shape}")

# Load NSR grid indices
idx_file = os.path.join(AMSR2, 'nsr_grid_idx.mat')
try:
    nsr_idx = hdf5storage.loadmat(idx_file)
except:
    nsr_idx = loadmat(idx_file)
print("NSR grid idx keys:", list(nsr_idx.keys()))
for k in nsr_idx.keys():
    if not k.startswith('_'):
        print(f"  {k}: shape={np.array(nsr_idx[k]).shape}")

# %% Cell 5: Define 7 segments along NSR
# Segment definitions (approximate waypoints)
segments = {
    'S1': {'name': 'Bering Strait - Pevek', 'lon_range': (170, -170), 'lat_range': (64, 71)},
    'S2': {'name': 'Pevek - New Siberian Is.', 'lon_range': (140, 170), 'lat_range': (69, 75)},
    'S3': {'name': 'New Siberian Is. - Vilkitsky', 'lon_range': (104, 140), 'lat_range': (74, 78)},
    'S4': {'name': 'Vilkitsky - Dikson', 'lon_range': (80, 104), 'lat_range': (72, 78)},
    'S5': {'name': 'Dikson - Novaya Zemlya', 'lon_range': (55, 80), 'lat_range': (70, 75)},
    'S6': {'name': 'Novaya Zemlya - Murmansk', 'lon_range': (33, 55), 'lat_range': (68, 73)},
    'S7': {'name': 'Murmansk - Norwegian Sea', 'lon_range': (5, 33), 'lat_range': (65, 70)},
}
print(f"Defined {len(segments)} segments")

# %% Cell 6: Read one AMSR2 file to check structure
sample_files = sorted(glob.glob(os.path.join(AMSR2, '2025', '*.nc')))
if sample_files:
    ds = xr.open_dataset(sample_files[0])
    print("Sample file variables:", list(ds.data_vars))
    print("Dimensions:", dict(ds.dims))
    print("Coords:", list(ds.coords))
    print(ds)
    ds.close()

# %% Cell 7: Process all AMSR2 daily SIC - extract along NSR route
# This is the main computation - processes ~4900 files
# Strategy: for each year/month, compute mean SIC per segment

years = range(2012, 2026)
months = range(1, 13)

# We'll store: segment x year x month mean SIC
seg_names = list(segments.keys())
results = np.full((len(seg_names), len(list(years)), 12), np.nan)
nav_days = np.full((len(seg_names), len(list(years)), 12), np.nan)

print("Processing AMSR2 daily SIC files...")
print("This may take 30-60 minutes with 59 cores and 239GB RAM")

for yi, year in enumerate(years):
    year_dir = os.path.join(AMSR2, str(year))
    if not os.path.exists(year_dir):
        print(f"  {year}: directory not found, skipping")
        continue
    
    files = sorted(glob.glob(os.path.join(year_dir, '*.nc')))
    if not files:
        print(f"  {year}: no files found")
        continue
    
    # Read all files for this year at once using xr.open_mfdataset
    try:
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', 
                                parallel=False, engine='netcdf4')
    except Exception as e:
        print(f"  {year}: mfdataset failed ({e}), trying file-by-file")
        # Fallback: read file by file
        daily_sic = []
        daily_dates = []
        for f in files:
            try:
                d = xr.open_dataset(f)
                # Extract date from filename
                fname = os.path.basename(f)
                datestr = fname.split('-')[3].split('-')[0] if 'n3125' in fname else fname[:8]
                # Try to parse date from filename: asi-AMSR2-n3125-YYYYMMDD-v5.4.nc
                parts = fname.split('-')
                for p in parts:
                    if len(p) == 8 and p.isdigit():
                        datestr = p
                        break
                
                # Get the SIC variable (try common names)
                sic_var = None
                for vname in d.data_vars:
                    if 'sic' in vname.lower() or 'ice' in vname.lower() or 'z' == vname.lower():
                        sic_var = vname
                        break
                if sic_var is None:
                    sic_var = list(d.data_vars)[0]
                
                sic_data = d[sic_var].values
                daily_sic.append(sic_data)
                daily_dates.append(datestr)
                d.close()
            except:
                continue
        
        if not daily_sic:
            print(f"  {year}: no valid files")
            continue
        
        daily_sic = np.array(daily_sic)
        
        # Process by month
        for mi in range(12):
            month = mi + 1
            month_mask = [i for i, d in enumerate(daily_dates) 
                         if len(d) >= 6 and int(d[4:6]) == month]
            
            if not month_mask:
                continue
            
            month_sic = daily_sic[month_mask]
            mean_sic = np.nanmean(month_sic, axis=0)
            
            # For each segment, extract SIC using lat/lon ranges
            for si, (seg_key, seg_info) in enumerate(segments.items()):
                lr = seg_info['lat_range']
                lonr = seg_info['lon_range']
                
                # Create mask based on lat/lon grid
                if lon_grid.shape == mean_sic.shape:
                    lat_mask = (lat_grid >= lr[0]) & (lat_grid <= lr[1])
                    if lonr[0] < lonr[1]:
                        lon_mask = (lon_grid >= lonr[0]) & (lon_grid <= lonr[1])
                    else:  # crosses dateline
                        lon_mask = (lon_grid >= lonr[0]) | (lon_grid <= lonr[1])
                    mask = lat_mask & lon_mask
                    
                    seg_sic = mean_sic[mask]
                    seg_sic = seg_sic[seg_sic >= 0]  # remove fill values
                    if len(seg_sic) > 0:
                        # Convert to percentage if needed (0-100 or 0-1)
                        if np.nanmax(seg_sic) > 1:
                            seg_sic = seg_sic / 100.0
                        results[si, yi, mi] = np.nanmean(seg_sic) * 100  # store as %
                        
                        # Count navigable days (SIC < 15%)
                        month_days = month_sic[:, mask[0] if mask.ndim > 1 else mask]
                        if np.nanmax(month_days) > 1:
                            month_days = month_days / 100.0
                        nav_count = np.sum(np.nanmean(month_days, axis=1) < 0.15)
                        nav_days[si, yi, mi] = nav_count
                else:
                    # If grid shape doesn't match, use fractional indices
                    ny, nx = mean_sic.shape
                    y_start = int(lr[0] / 90 * ny)
                    y_end = int(lr[1] / 90 * ny)
                    results[si, yi, mi] = np.nanmean(mean_sic) * 100
        
        continue
    
    # If mfdataset worked
    print(f"  {year}: loaded {len(files)} files")

print("AMSR2 processing complete!")
print(f"Results shape: {results.shape}")

# Save intermediate results
np.save(os.path.join(OUT, 'seg_monthly_sic.npy'), results)
np.save(os.path.join(OUT, 'seg_monthly_navdays.npy'), nav_days)
print("Saved intermediate results")

# %% Cell 8: Figure 3a - Monthly mean SIC heatmap by segment
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 3a: Climatological monthly SIC by segment
clim_sic = np.nanmean(results, axis=1)  # average over years
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

im = axes[0].imshow(clim_sic, aspect='auto', cmap='Blues_r', vmin=0, vmax=100)
axes[0].set_yticks(range(len(seg_names)))
axes[0].set_yticklabels(seg_names)
axes[0].set_xticks(range(12))
axes[0].set_xticklabels(month_labels, rotation=45)
axes[0].set_title('(a) Monthly Mean SIC by Segment (2012–2025)', fontsize=12)
plt.colorbar(im, ax=axes[0], label='SIC (%)')

# Add text values
for i in range(len(seg_names)):
    for j in range(12):
        val = clim_sic[i, j]
        if not np.isnan(val):
            color = 'white' if val > 50 else 'black'
            axes[0].text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, color=color)

# 3b: Annual navigable window time series
year_list = list(range(2012, 2026))
# Whole route navigable days per year (minimum across all segments)
annual_nav = []
for yi in range(len(year_list)):
    yearly_total = 0
    for mi in range(12):
        # All segments must be navigable
        seg_nav = nav_days[:, yi, mi]
        if np.all(~np.isnan(seg_nav)):
            yearly_total += np.min(seg_nav)
        elif np.any(~np.isnan(seg_nav)):
            yearly_total += np.nanmin(seg_nav)
    annual_nav.append(yearly_total)

axes[1].bar(year_list, annual_nav, color='steelblue', alpha=0.7)
# Add trend line
valid = [(y, n) for y, n in zip(year_list, annual_nav) if n > 0]
if len(valid) > 2:
    vx, vy = zip(*valid)
    z = np.polyfit(vx, vy, 1)
    p = np.poly1d(z)
    axes[1].plot(vx, p(vx), 'r--', linewidth=2, label=f'Trend: {z[0]:.1f} days/yr')
    axes[1].legend()

axes[1].set_xlabel('Year')
axes[1].set_ylabel('Navigable Days (SIC < 15%, all segments)')
axes[1].set_title('(b) Annual Navigable Window (2012–2025)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Fig3_SIC_analysis.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'Fig3_SIC_analysis.tiff'), dpi=300, bbox_inches='tight')
print("Figure 3 saved!")
plt.show()

# %% Cell 9: Figure 1 - Study Area Map (simplified without cartopy)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot September mean SIC
sep_sic = results[:, :, 8]  # September = index 8
sep_clim = np.nanmean(sep_sic, axis=1)

# NSR route waypoints
nsr_lons = [129.0, 142.0, -173.2, -169.0, 170.3, 140.0, 104.3, 80.4, 55.0, 33.08, 5.0, 4.5]
nsr_lats = [35.1, 45.8, 64.4, 66.0, 69.7, 74.0, 77.7, 73.5, 72.0, 68.97, 65.0, 51.9]

# SCR route waypoints  
scr_lons = [129.0, 119.0, 103.8, 100.0, 80.0, 45.0, 32.58, 32.34, -5.5, 0.0, 4.5]
scr_lats = [35.1, 24.0, 1.2, -1.0, 5.5, 12.5, 29.95, 31.27, 36.0, 50.0, 51.9]

ax.plot(nsr_lons, nsr_lats, 'b-o', linewidth=2, markersize=5, label='NSR (~7,400 nm)', zorder=5)
ax.plot(scr_lons, scr_lats, 'r--s', linewidth=2, markersize=5, label='SCR (~11,200 nm)', zorder=5)

# Mark segments
seg_mids = {
    'S1': (180, 68), 'S2': (155, 72), 'S3': (122, 76),
    'S4': (92, 75), 'S5': (67, 72.5), 'S6': (44, 70.5), 'S7': (19, 67.5)
}
for seg, (x, y) in seg_mids.items():
    ax.annotate(seg, (x, y), fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

# Mark key points
key_points = {
    'Busan': (129.0, 35.1), 'Rotterdam': (4.5, 51.9),
    'Bering Strait': (-169.0, 66.0), 'Vilkitsky Strait': (104.3, 77.7),
    'Suez Canal': (32.5, 30.5)
}
for name, (x, y) in key_points.items():
    ax.annotate(name, (x, y), fontsize=8, fontweight='bold',
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xlim(-180, 180)
ax.set_ylim(-10, 85)
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title('Figure 1: Study Area - NSR and SCR Routes with Segment Definitions', fontsize=13)
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Fig1_study_area.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'Fig1_study_area.tiff'), dpi=300, bbox_inches='tight')
print("Figure 1 saved!")
plt.show()

# %% Cell 10: Print summary statistics for Table 3
print("\n=== Table 3: Mean Navigable Days by Segment and Period ===")
print(f"{'Segment':<10} {'2012-2016':>10} {'2017-2021':>10} {'2021-2025':>10}")
print("-" * 45)
for si, seg in enumerate(seg_names):
    p1 = np.nansum(nav_days[si, 0:5, :])  / 5  # 2012-2016
    p2 = np.nansum(nav_days[si, 5:10, :]) / 5  # 2017-2021
    p3 = np.nansum(nav_days[si, 9:14, :]) / 5  # 2021-2025
    print(f"{seg:<10} {p1:>10.0f} {p2:>10.0f} {p3:>10.0f}")

print("\nPart 1 complete! Proceed to Part 2.")
