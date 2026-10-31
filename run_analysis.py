
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "catchments_clean.csv"
OUT = ROOT / "outputs" / "results"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INPUT)

# DEMONSTRATION ONLY: provisional Curve Numbers. Soil names are not verified
# hydrologic soil groups, so these mappings must be replaced with local HSG data.
soil_hsg_demo = {"Sandy Loam": "B", "Loam": "B", "Silty": "C", "Clay": "D"}
cn_table_demo = {
    "Built-up": {"B": 85, "C": 90, "D": 92},
    "Road": {"B": 92, "C": 94, "D": 95},
    "Open Soil": {"B": 80, "C": 86, "D": 89},
    "Park/Green": {"B": 61, "C": 74, "D": 80},
    "Waterbody": {"B": 98, "C": 98, "D": 98},
}
df["demo_hsg_assumption"] = df["soil_type"].map(soil_hsg_demo)
df["demo_curve_number"] = df.apply(
    lambda r: cn_table_demo.get(r["land_use"], {}).get(r["demo_hsg_assumption"], np.nan),
    axis=1
)

# SCS-CN event runoff depth: Q=(P-0.2S)^2/(P+0.8S), if P>0.2S;
# S=25400/CN-254 (mm). Here rainfall intensity is treated as a 1-hour
# rainfall depth solely for this demo. It is NOT a calibrated event model.
P = df["rainfall_intensity_mm_hr"].clip(lower=0)
S = 25400 / df["demo_curve_number"] - 254
Ia = 0.2 * S
Q = np.where(P > Ia, ((P - Ia) ** 2) / (P + 0.8*S), 0)
df["scs_cn_runoff_depth_mm_demo"] = Q
df["scs_cn_runoff_volume_m3_demo"] = Q / 1000 * df["area_ha"] * 10000

# Transparent weighted-sum MCDA. Each criterion scaled 0..1.
# Higher rainfall, slope, imperviousness and poor drainage => higher priority.
def minmax(s):
    lo, hi = s.min(), s.max()
    return (s-lo)/(hi-lo) if hi != lo else pd.Series(0.0, index=s.index)

df["score_rain"] = minmax(df["rainfall_intensity_mm_hr"])
df["score_slope"] = minmax(df["slope_percent"])
df["score_impervious"] = minmax(df["impervious_surface_percent"])
df["score_drainage"] = df["drainage_capacity_class"].map(
    {"High": 0.0, "Medium": 0.5, "Low": 1.0}
).fillna(0.5)
# Weights sum to 1; illustrative, not stakeholder-validated.
weights = {"score_rain": 0.35, "score_slope": 0.20,
           "score_impervious": 0.30, "score_drainage": 0.15}
df["mcda_priority_score_demo"] = sum(df[k]*w for k, w in weights.items())
df["mcda_priority_rank_demo"] = df["mcda_priority_score_demo"].rank(
    ascending=False, method="min"
).astype(int)
df["mcda_priority_class_demo"] = pd.cut(
    df["mcda_priority_score_demo"], bins=[-0.001, 0.33, 0.66, 1.001],
    labels=["Low", "Medium", "High"]
).astype(str)

df.to_csv(OUT/"catchment_analysis_results.csv", index=False)
print("Saved:", OUT/"catchment_analysis_results.csv")
print("Top 5 MCDA priority catchments:")
print(df.sort_values("mcda_priority_score_demo", ascending=False)[
    ["catchment_id","mcda_priority_score_demo","mcda_priority_class_demo"]
].head().to_string(index=False))
print("NOTE: SCS-CN soil/CN mappings and MCDA weights are provisional demo assumptions.")
