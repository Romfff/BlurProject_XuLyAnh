import cv2
import numpy as np
from pathlib import Path

def load_dnn(prototxt_path, model_path):
    if not Path(prototxt_path).exists() or not Path(model_path).exists():
        raise FileNotFoundError("DNN model file(s) not found.")
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_faces_dnn(image, net, conf_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes
