"""Segmentation mask area -> volume -> mass -> nutrition.

Assumes a top-down photo with a reference object of known size on the same plane
(e.g. a plate of known diameter). Height comes from a per-food prior unless a
measured/depth-derived height is passed in. Every result carries a low/high range.
"""
import csv
from dataclasses import dataclass, field

NUTRIENTS = ["kcal", "protein_g", "carb_g", "fat_g", "sodium_mg"]


class FoodDB:
    def __init__(self, path):
        self.rows = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in NUTRIENTS + ["density_g_ml", "height_cm", "shape_factor"]:
                    r[k] = float(r[k])
                self.rows[r["label"]] = r

    def __getitem__(self, label):
        if label not in self.rows:
            raise KeyError(f"'{label}' is not in the nutrition database")
        return self.rows[label]


def px_per_cm_from_reference(ref_px, ref_cm):
    """ref_px: measured size of the reference in pixels; ref_cm: its real size."""
    if ref_px <= 0 or ref_cm <= 0:
        raise ValueError("reference sizes must be positive")
    return ref_px / ref_cm


@dataclass
class ItemEstimate:
    label: str
    area_cm2: float
    volume_ml: float
    mass_g: float
    mass_low_g: float
    mass_high_g: float
    nutrition: dict = field(default_factory=dict)
    nutrition_low: dict = field(default_factory=dict)
    nutrition_high: dict = field(default_factory=dict)


def _scale(row, grams):
    return {k: row[k] * grams / 100.0 for k in NUTRIENTS}


def estimate_item(label, mask_area_px, px_per_cm, db, height_cm=None, uncertainty=0.25):
    row = db[label]
    area_cm2 = mask_area_px / (px_per_cm ** 2)
    h = height_cm if height_cm is not None else row["height_cm"]
    volume = area_cm2 * h * row["shape_factor"]  # 1 cm3 == 1 ml
    mass = volume * row["density_g_ml"]
    lo, hi = mass * (1 - uncertainty), mass * (1 + uncertainty)
    return ItemEstimate(label, area_cm2, volume, mass, lo, hi,
                        _scale(row, mass), _scale(row, lo), _scale(row, hi))


def estimate_meal(items, px_per_cm, db, uncertainty=0.25):
    """items: list of (label, mask_area_px) or (label, mask_area_px, height_cm)."""
    results = []
    total, lo, hi = (dict.fromkeys(NUTRIENTS, 0.0) for _ in range(3))
    for it in items:
        h = it[2] if len(it) > 2 else None
        r = estimate_item(it[0], it[1], px_per_cm, db, h, uncertainty)
        results.append(r)
        for k in NUTRIENTS:
            total[k] += r.nutrition[k]; lo[k] += r.nutrition_low[k]; hi[k] += r.nutrition_high[k]
    return {"items": results, "total": total, "low": lo, "high": hi}
