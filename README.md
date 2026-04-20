# NSR Sustainability Index: Data and Code Repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19658577.svg)](https://doi.org/10.5281/zenodo.19658577)

## Overview

This repository contains the processed datasets and analysis code for the following publication:

**"Toward Sustainable Arctic Shipping: A Geospatial Framework for Optimizing Carbon Benefits While Minimizing Environmental Costs Along the Northern Sea Route"**

Lee, S.-J., Kim, J., & Yoo, H.-S.

*Sustainability Science* (submitted 2026)

## Repository Structure
├── data/ │ ├── seg_monthly_sic.npy # Monthly mean SIC (%) by segment, year, month [7×14×12] │ ├── seg_monthly_navdays.npy # Navigable days (SIC<15%) by segment, year, month [7×14×12] │ ├── cbr_segment.npy # Segment-level Carbon Benefit Ratio (%) [7×12] │ ├── cbr_total.npy # Total route Carbon Benefit Ratio (%) [12] │ ├── nsr_co2_monthly.npy # NSR CO₂ emissions by segment and month (tonnes) [7×12] │ ├── ess_scores.npy # Ecological Sensitivity Score by segment [7] │ ├── TableS1_monthly_SIC.csv # Complete monthly SIC for all segments and years (Table S1) │ ├── TableS3_AHP_ESS.csv # AHP pairwise comparison matrix for ESS (Table S3) │ └── TableS4_SI_values.csv # Sustainability Index values for all segment-months (Table S4) ├── code/ │ ├── part1_sic_analysis.py # AMSR2 SIC processing, Figure 1, Figure 3 │ ├── part2_emissions_ecology.py # CO₂ emissions, ESS calculation, Figure 4, Figure 5 │ └── part3_sustainability_index.py # SI computation, Figure 6, sensitivity analysis, Figure S1-S3 └── README.md

## Data Description

### Processed Data Files (.npy)

| File | Shape | Description |
|------|-------|-------------|
| `seg_monthly_sic.npy` | (7, 14, 12) | Monthly mean sea ice concentration (%) for 7 NSR segments (S1–S7), 14 years (2012–2025), 12 months |
| `seg_monthly_navdays.npy` | (7, 14, 12) | Number of navigable days (daily SIC < 15%) per segment, year, and month |
| `cbr_segment.npy` | (7, 12) | Segment-level Carbon Benefit Ratio (%) relative to SCR, climatological average |
| `cbr_total.npy` | (12,) | Whole-route Carbon Benefit Ratio (%) by month |
| `nsr_co2_monthly.npy` | (7, 12) | NSR CO₂ emissions (tonnes) by segment and month for a 6,500 TEU reference vessel |
| `ess_scores.npy` | (7,) | Composite Ecological Sensitivity Score (0–1) per segment |

### Supplementary Tables (.csv)

| File | Description |
|------|-------------|
| `TableS1_monthly_SIC.csv` | Complete monthly mean SIC (%) for all 7 segments × 14 years (2012–2025) |
| `TableS3_AHP_ESS.csv` | AHP pairwise comparison matrix and derived weights for ESS sub-criteria |
| `TableS4_SI_values.csv` | Sustainability Index values for all 84 segment-month combinations |

### NSR Segment Definitions

| Segment | From | To | Distance (nm) |
|---------|------|----|---------------|
| S1 | Bering Strait | Pevek | 500 |
| S2 | Pevek | New Siberian Islands | 800 |
| S3 | New Siberian Islands | Vilkitsky Strait | 800 |
| S4 | Vilkitsky Strait | Dikson | 500 |
| S5 | Dikson | Novaya Zemlya | 600 |
| S6 | Novaya Zemlya | Murmansk | 800 |
| S7 | Murmansk | Norwegian Sea | 600 |

### Reference Vessel Parameters

- **Vessel type:** 6,500 TEU container ship
- **Main engine power:** 42,000 kW
- **SFOC:** 175 g/kWh
- **Operational speed:** 16 knots
- **Fuel type:** HFO (CF = 3.114 g CO₂/g fuel)
- **Route:** Busan, South Korea → Rotterdam, Netherlands

## Original Data Sources

The raw input datasets used in this study are publicly available:

- **AMSR2 Sea Ice Concentration** (3.125 km, daily, 2012–2025): University of Bremen — https://seaice.uni-bremen.de/sea-ice-concentration/amsre-amsr2/
- **ERA5 Climate Reanalysis** (0.25°, monthly): Copernicus Climate Data Store — https://cds.climate.copernicus.eu/
- **Marine Protected Areas**: World Database on Protected Areas — https://www.protectedplanet.net/
- **Marine Mammal Occurrences**: Ocean Biodiversity Information System — https://obis.org/
- **Bathymetry**: GEBCO 2024 — https://www.gebco.net/ and IBCAO v4

## Software Requirements

- Python ≥ 3.9
- NumPy, Pandas, Matplotlib, SciPy, xarray, netCDF4, h5py, hdf5storage

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy xarray netcdf4 h5py hdf5storage

How to Load Data
import numpy as np
import pandas as pd

# Load processed arrays
sic = np.load('data/seg_monthly_sic.npy')         # shape: (7, 14, 12)
cbr = np.load('data/cbr_segment.npy')              # shape: (7, 12)
ess = np.load('data/ess_scores.npy')               # shape: (7,)
si_values = pd.read_csv('data/TableS4_SI_values.csv')

print(f"September mean SIC for S1: {np.nanmean(sic[0, :, 8]):.1f}%")
print(f"ESS for S1 (Bering Strait): {ess[0]:.2f}")

License
This dataset is released under CC BY 4.0.

Funding
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (No. RS-2026-25488422).

Contact
For questions regarding this dataset, please contact the corresponding author.
