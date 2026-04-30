# services/detector.py
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/best.pt")


class ShelfEmptySpaceDetector:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл весов модели не найден: {model_path.resolve()}"
            )

        self.model = YOLO(str(model_path))

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 1280,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список детекций в формате:
        {
            "label": "empty_space",
            "confidence": 0.91,
            "bbox": [x1, y1, x2, y2]
        }
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)

        results = self.model.predict(
            source=image_np,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "label": names.get(cls_id, str(cls_id)),
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return detections