"""Fine-tune, evaluate and export the food detector/segmenter. Run on Colab/Kaggle (free GPU).

  pip install ultralytics onnx onnxslim
  python training/train.py --epochs 80
"""
import argparse, json, shutil
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", default="training/data.yaml")
p.add_argument("--model", default="yolo11n-seg.pt")  # n = smallest, runs in a browser
p.add_argument("--epochs", type=int, default=80)
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, patience=20, project="runs", name="nutrigo")

best = YOLO("runs/nutrigo/weights/best.pt")
m = best.val(data=a.data, split="test", imgsz=a.imgsz)
report = {
    "box_mAP50": m.box.map50, "box_mAP50_95": m.box.map,
    "box_precision": float(m.box.mp), "box_recall": float(m.box.mr),
    "mask_mAP50": m.seg.map50, "mask_mAP50_95": m.seg.map,
    "per_class_mask_mAP50_95": {best.names[i]: float(v) for i, v in zip(m.seg.ap_class_index, m.seg.maps[m.seg.ap_class_index])},
}
json.dump(report, open("evaluation/detection_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))

onnx_path = best.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True)
shutil.copy(onnx_path, "web/model/best.onnx")
json.dump(list(best.names.values()), open("web/model/classes.json", "w"))
print("Exported web/model/best.onnx - commit it to the repo.")
