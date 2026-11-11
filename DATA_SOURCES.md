# Data sources and what is real

| File | What it is | Real or synthetic |
|---|---|---|
| `Pakistan_flood_dataset.csv` | Yearly flood and climate records for 17 areas, 2000 to 2024 | Origin not documented. Not checked against official reports; 2010 and 2022 do not stand out. Treat as demonstration data. |
| `pakistan_asset_recovery_2025_prototype.csv` | 500 damaged assets with repair costs and resource needs | Synthetic, calibrated to NDMA 2025 infrastructure damage information |
| `data_raw/pakistan_population_statistics_2017_...xlsx` | 2017 census population by province, district and tehsil (preliminary release, via OCHA) | Real. Excludes Gilgit-Baltistan and Azad Kashmir |
| `data_raw/pak_adminboundaries_tabulardata.xlsx` | Administrative boundary tables with area in km2 (OCHA) | Real. No coordinates |
| `data_raw/rainfall_1901_2016_pak.csv` | Monthly national rainfall, 1901 to 2016 | Real. One national series, not by district |
| `data_clean/pakistan_flood_impact_province_2020_2026.csv` | Province-level flood deaths, injuries, houses and people affected, 2022 to 2026 | Real, from NDMA and partner reports. Every row has a source link and date. 2026 is a partial season |
| `data_clean/district_flood_impact_sourced.csv` | District-level impact for KP (April 2024), Punjab 2022 and Punjab 2025 | Real, from PDMA reports. Partial coverage; blank means not reported |

Everything else in `data_clean/` is produced from the files above by `data_clean/clean_all.py`
(run it from the repository root). `data_clean/data_quality_log.csv` lists every check and what was done.
Problems are flagged in extra columns; no value is silently changed or deleted.

`app_districts_reference.csv` and `rainfall_scenario_multipliers.csv` sit in the top folder because the app reads them.

Not available from public sources: district-level flood records for most years, per-asset repair data,
district coordinates, daily rainfall or river levels, and actual stock of repair workers and equipment.
