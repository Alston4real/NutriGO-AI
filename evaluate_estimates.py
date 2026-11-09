"""Compare predicted vs measured volume, weight and nutrition.

CSV columns (one row per food item per photo):
  image_id,label,true_volume_ml,true_mass_g,true_kcal,pred_volume_ml,pred_mass_g,pred_kcal[,mass_low_g,mass_high_g]
Usage: python evaluation/evaluate_estimates.py results.csv
"""
import csv, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from metrics import mae, mape, coverage


def main(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    col = lambda k: [float(r[k]) for r in rows]
    print(f"items evaluated: {len(rows)}")
    for name, t, p in [("volume (ml)", "true_volume_ml", "pred_volume_ml"),
                       ("weight (g)", "true_mass_g", "pred_mass_g"),
                       ("energy (kcal)", "true_kcal", "pred_kcal")]:
        print(f"{name:14s} MAE={mae(col(p), col(t)):8.2f}   MAPE={mape(col(p), col(t)):6.1f}%")
    if rows and "mass_low_g" in rows[0]:
        print(f"weight range coverage: {coverage(col('mass_low_g'), col('mass_high_g'), col('true_mass_g'))*100:.0f}%")
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    print("\nper food MAPE (weight):")
    for l, rs in sorted(by.items()):
        print(f"  {l:26s} {mape([float(x['pred_mass_g']) for x in rs],[float(x['true_mass_g']) for x in rs]):6.1f}%  (n={len(rs)})")


if __name__ == "__main__":
    main(sys.argv[1])
