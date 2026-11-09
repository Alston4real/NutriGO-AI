# NutriGO AI

Photo -> detect + segment foods -> area -> volume -> mass -> nutrition, all free.

## Pipeline
1. `training/train.py` fine-tunes YOLO11-seg (Colab/Kaggle free GPU), writes `evaluation/detection_report.json` (mAP, precision, recall, mask mAP) and exports `web/model/best.onnx`.
2. `web/index.html` runs the model in the browser (ONNX Runtime Web). The plate (known diameter) gives the cm-per-pixel scale. Mask area x food height prior x shape factor x density = grams. grams/100 x per-100g values = nutrition, with a +/-25% range.
3. `nutrinogo/estimator.py` is the same maths in Python for evaluation against your scale and volume measurements.
4. `evaluation/evaluate_estimates.py` reports MAE and MAPE for volume, weight and energy.

## Do this next
1. Collect ~100 top-down photos per dish with a plate in frame. Weigh each component and write the grams down. Label in Roboflow (free) using the classes in `training/data.yaml`.
2. Put the export in `data/`, then in Colab: `pip install ultralytics onnx onnxslim && python training/train.py`.
3. Commit `web/model/best.onnx` and `web/model/classes.json`; enable GitHub Pages on `/web`.
4. Replace every `PLACEHOLDER_verify_*` row in `nutrition_db/foods.csv` with HPB/USDA values, and calibrate height/shape/density from your measurements.
5. Run `python tests/test_estimator.py` after any change.

Split train/val/test by meal session, not by image, or scores will be inflated.
