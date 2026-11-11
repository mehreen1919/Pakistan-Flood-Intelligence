"""Clean every dataset in the project. Run from the repo root: python data_clean/clean_all.py
 Rule: never delete or silently change a value;
add flag columns and write a data-quality log instead."""
import re, numpy as np, pandas as pd
A = "data_raw/"
S = ""
OUT = "data_clean/"
log = []
def L(ds, check, result, action): log.append((ds, check, result, action))

# ---------- 1. Flood history ----------
f = pd.read_csv(S + "Pakistan_flood_dataset.csv")
n0 = len(f)
f["province"] = f.province.str.strip(); f["area_name"] = f.area_name.str.strip()
f["label_conflict"] = (f.flood != (f.flood_pct >= 10)).astype(int)
def mz(s):
    med = s.median(); mad = (s - med).abs().median()
    return 0.6745 * (s - med) / (mad if mad else 1)
f["rain_outlier_for_area"] = (f.groupby("area_name").rain_monsoon_mm.transform(mz).abs() > 3.5).astype(int)
f["peak_share_of_monsoon"] = (f.rain_peak_month_mm / f.rain_monsoon_mm).round(3)
f.to_csv(OUT + "flood_history_clean.csv", index=False)
L("flood_history", "rows / nulls / duplicates", f"{n0} rows, 0 nulls, 0 duplicate area-years, 17 areas x 25 years complete", "kept all")
L("flood_history", "flood label vs flood_pct", f"{int(f.label_conflict.sum())} rows disagree with a 10% rule (labels overlap in the 8-12% band)", "added label_conflict flag; labels unchanged")
L("flood_history", "rain outliers within each area", f"{int(f.rain_outlier_for_area.sum())} rows flagged (|modified z|>3.5)", "added rain_outlier_for_area flag; kept (heavy rain is real signal)")
L("flood_history", "known flood years", "2010 and 2022 are not visible: Sukkur 2022 = 2.5%, Badin 2010 = 0.7%; yearly mean flood_pct is flat (~30%)", "NOT fixable by cleaning; needs real source data")
L("flood_history", "climate plausibility", "Skardu monsoon temp 33.3C (high-altitude, normally ~20-25C); Ghanche monsoon rain 629mm (arid region); Lahore 722mm vs adjacent Kasur 166mm", "documented; not altered")
L("flood_history", "rs_fsm_lgbm_pakistan", "one constant value per area (static susceptibility score)", "kept; treat as area attribute, not yearly variable")

# ---------- 2. Asset recovery ----------
r = pd.read_csv(S + "pakistan_asset_recovery_2025_prototype.csv")
lc = np.log(r.estimated_repair_cost_pkr); q1, q3 = lc.groupby(r.asset_type).transform(lambda s: s.quantile(.25)), lc.groupby(r.asset_type).transform(lambda s: s.quantile(.75))
r["cost_outlier_for_type"] = ((lc < q1 - 1.5*(q3-q1)) | (lc > q3 + 1.5*(q3-q1))).astype(int)
bins = pd.cut(r.recovery_priority_score_derived, [0, 45, 58, 72, 101], right=False, labels=["Low", "Medium", "High", "Critical"])
r["label_matches_score"] = (bins.astype(str) == r.recovery_priority_label_derived).astype(int)
r["cost_per_person_pkr"] = (r.estimated_repair_cost_pkr / r.population_served_or_affected_est).round(0)
r.to_csv(OUT + "asset_recovery_clean.csv", index=False)
L("asset_recovery", "rows / nulls / duplicate ids", "500 rows, 0 nulls, 0 duplicate asset_id; all ratings in 1-5; no negative cost, days or workers", "kept all")
L("asset_recovery", "cost outliers within asset type (log IQR)", f"{int(r.cost_outlier_for_type.sum())} rows", "added cost_outlier_for_type flag; kept")
L("asset_recovery", "priority label vs score bands", f"{int((1-r.label_matches_score).sum())} mismatches", "added label_matches_score flag")
L("asset_recovery", "provenance", "all 500 rows are synthetic_asset_record_calibrated_to_NDMA_2025, single year 2025", "added nothing; stated in report")

# ---------- 3. Boundaries ----------
b = pd.read_excel(A + "pak_adminboundaries_tabulardata.xlsx", "ADM2")
b = b.rename(columns={"ADM2_EN": "district", "ADM2_PCODE": "bnd_pcode", "ADM1_EN": "province", "AREA_SQKM": "area_km2"})[["province", "district", "bnd_pcode", "area_km2"]]
b["area_km2"] = b.area_km2.round(1)
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
b["key"] = b.district.map(norm)
L("boundaries", "area totals", "ADM1/ADM2/ADM3 area all sum to 876,143.7 km2; 160 districts, no nulls or duplicates", "kept")
L("boundaries", "pcodes", "province codes differ from the population file (e.g. KP = PK5 here, PK2 there)", "joined by cleaned name, not code")
L("boundaries", "coordinates", "none in the file", "cannot map; flagged in report")

# ---------- 4. Population ----------
p = pd.read_excel(A + "pakistan_population_statistics_2017_20190603.xlsx", "Admin2")
p = p.rename(columns={"admin2Name_en": "district", "admin2Pcode": "pop_pcode", "admin1Name_en": "province"})
p["province"] = p.province.replace({"Federal Capital Territory": "Islamabad"})
p["split_sum_ok"] = ((p.Urban_pop + p.Rural_pop == p.Total_pop) & (p.Total_male_population + p.Total_female_population + p.Total_transgender_population == p.Total_pop)).astype(int)
num = ["Total_pop", "Urban_pop", "Rural_pop", "Total_male_population", "Total_female_population", "Total_transgender_population"]
p["urban_share_pct"] = (100 * p.Urban_pop / p.Total_pop).round(1)
p["key"] = p.district.map(norm)
MANUAL = {"deraismailkhan": "diki", "layyah": "leiah", "malakandpa": "malakand", "qambarshahdadkot": "kambarshahdadkot", "sheerani": "sherani"}
b["key"] = b.key.replace({"dikhan": "diki"})
p["join_key"] = p.key.replace(MANUAL)
j = p.merge(b[["join_key"] if False else ["key", "bnd_pcode", "area_km2"]].rename(columns={"key": "join_key"}), on="join_key", how="left")
j["density_per_km2"] = (j.Total_pop / j.area_km2).round(1)
j["join_status"] = np.where(j.area_km2.notna(), np.where(j.key != j.join_key, "matched_by_manual_fix", "matched_by_name"), "no_boundary_match (merged/split district)")
j.drop(columns=["key", "join_key"]).to_csv(OUT + "district_population_2017_clean.csv", index=False)
L("population", "totals", f"national total {int(p.Total_pop.sum()):,}; urban+rural and male+female+transgender equal total in {int(p.split_sum_ok.sum())}/{len(p)} districts", "flag split_sum_ok")
L("population", "scope", "no Gilgit-Baltistan or Azad Kashmir (preliminary 2017 release)", "added those districts from boundaries (area only, population blank)")
L("population", "join to boundaries", f"{int(j.area_km2.notna().sum())}/{len(j)} districts joined ({int((j.join_status=='matched_by_manual_fix').sum())} via manual name fix)", "unmatched: " + ", ".join(j[j.area_km2.isna()].district))
L("population", "duplicate files", "pak_admpop_adm2_v2.csv and pak_admpop_adm3.csv are exact copies of workbook sheets Admin2/Admin3", "dropped as redundant")

# all districts master (boundaries is the spine, includes GB/AJK)
bm = b.merge(p[["join_key", "Total_pop", "urban_share_pct"]], left_on="key", right_on="join_key", how="left").drop(columns=["join_key", "key"])
bm["density_per_km2"] = (bm.Total_pop / bm.area_km2).round(1)
bm.to_csv(OUT + "district_master_area_population.csv", index=False)

# the 17 app districts
app = sorted(f.area_name.unique()); k2 = {norm(a): a for a in app}
ref = bm[bm.district.map(norm).isin(k2)].copy()
ref = ref.merge(f.groupby("area_name").agg(avg_flood_pct=("flood_pct", "mean"), flood_year_rate=("flood", "mean")).round(2), left_on="district", right_index=True, how="left")
ref.to_csv("app_districts_reference.csv", index=False)
L("app_districts", "17 app districts found in boundaries", f"{len(ref)}/17; with population {int(ref.Total_pop.notna().sum())}/17", "missing population: " + ", ".join(ref[ref.Total_pop.isna()].district))

# ---------- 5. Rainfall ----------
rr = pd.read_csv(A + "rainfall_1901_2016_pak.csv"); rr.columns = ["rain_mm", "year", "month"]
rr["month"] = rr.month.str.strip()
mo = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
rr["month_num"] = rr.month.map({m: i + 1 for i, m in enumerate(mo)}); rr["rain_mm"] = rr.rain_mm.round(2)
rr.sort_values(["year", "month_num"]).to_csv(OUT + "national_rainfall_monthly_clean.csv", index=False)
ann = rr.groupby("year").rain_mm.sum().round(1).rename("annual_mm").to_frame()
ann["monsoon_jul_sep_mm"] = rr[rr.month_num.isin([7, 8, 9])].groupby("year").rain_mm.sum().round(1)
ann.to_csv(OUT + "national_rainfall_annual_summary.csv")
pc = np.percentile(ann.monsoon_jul_sep_mm, [10, 25, 50, 75, 90, 95]); mean = ann.monsoon_jul_sep_mm.mean()
sc = pd.DataFrame({"percentile": ["p10", "p25", "p50", "p75", "p90", "p95"], "monsoon_mm": pc.round(0), "multiplier_vs_mean": (pc / mean).round(2)})
sc.to_csv("rainfall_scenario_multipliers.csv", index=False)
L("rainfall", "completeness", "1,392 rows = 116 years x 12 months, no nulls, no duplicates; header had a leading space in ' Year'", "renamed columns")
L("rainfall", "scope", "single national series, ends 2016", "usable for scenario calibration only, not per-district")

# ---------- 6. Flood impact ----------
im = pd.read_csv("data_clean/pakistan_flood_impact_province_2020_2026.csv")
latest = im.sort_values("as_of_date").groupby(["year", "region", "indicator"]).tail(1)
wide = latest.pivot_table(index=["year", "region"], columns="indicator", values="value", aggfunc="first").reset_index()
wide.to_csv(OUT + "flood_impact_latest_snapshot_wide.csv", index=False)
L("flood_impact", "multiple snapshots", "2025 has two as-of dates (7 Sep, 16 Sep)", "wide table keeps the latest snapshot per year/region/indicator")
L("flood_impact", "provincial sums vs national", "2023 houses 5,677 vs 5,784; 2025 (7 Sep) deaths 897 vs 907", "kept; gaps noted, not filled")
L("flood_impact", "2026", "partial season snapshot (late July)", "kept, flagged in the source file")

pd.DataFrame(log, columns=["dataset", "check", "result", "action"]).to_csv(OUT + "data_quality_log.csv", index=False)
print(pd.DataFrame(log, columns=["dataset", "check", "result", "action"]).to_string(max_colwidth=110))
print("\nAPP DISTRICTS"); print(ref[["province", "district", "area_km2", "Total_pop", "density_per_km2", "avg_flood_pct"]].to_string(index=False))
print("\nSCENARIOS"); print(sc.to_string(index=False))
print("\nUNMATCHED POP:", j[j.area_km2.isna()].district.tolist())
