import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
import numpy as np
from nutrinogo import FoodDB, estimate_item, estimate_meal, px_per_cm_from_reference
from metrics import iou, miou, mape, mae, topk_accuracy, coverage

db = FoodDB(os.path.join(os.path.dirname(__file__), "..", "nutrition_db", "foods.csv"))

def test_scale():
    assert px_per_cm_from_reference(440, 22) == 20

def test_known_geometry():
    r = estimate_item("chicken_rice_rice", 40000, 20, db)  # 40000 px at 20 px/cm = 100 cm2
    assert math.isclose(r.area_cm2, 100)
    assert math.isclose(r.volume_ml, 100 * 2.5 * 0.70)
    assert math.isclose(r.mass_g, 100 * 2.5 * 0.70 * 0.80)
    assert math.isclose(r.nutrition["kcal"], 175 * r.mass_g / 100)
    assert r.mass_low_g < r.mass_g < r.mass_high_g

def test_meal_sum_and_range():
    m = estimate_meal([("chicken_rice_rice", 40000), ("roast_chicken", 20000, 2.0)], 20, db)
    assert math.isclose(m["total"]["kcal"], sum(i.nutrition["kcal"] for i in m["items"]))
    assert m["low"]["kcal"] < m["total"]["kcal"] < m["high"]["kcal"]

def test_unknown_food():
    try:
        estimate_item("pizza", 100, 20, db); assert False
    except KeyError:
        pass

def test_metrics():
    a = np.zeros((10, 10)); a[:5] = 1
    b = np.zeros((10, 10)); b[:5, :5] = 1
    assert math.isclose(iou(a, b), 0.5)
    assert math.isclose(miou([a, b], [a, a], ["x", "y"])[0], 0.75)
    assert mape([110], [100]) == 10 and mae([110], [100]) == 10
    assert topk_accuracy([[.1, .9, 0], [.5, .3, .2]], [1, 1], k=1) == 0.5
    assert topk_accuracy([[.1, .9, 0], [.5, .3, .2]], [1, 1], k=2) == 1.0
    assert coverage([90], [120], [100]) == 1.0

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
