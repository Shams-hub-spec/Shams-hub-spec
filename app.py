from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

# ---------------------------
# HEALTH CHECK ROUTE
# ---------------------------
@app.route("/health")
def health():
    return {"status": "ok"}


# ---------------------------
# YOLO MODELNI YUKLASH
# ---------------------------
YOLO_CONFIG = "yolov3.cfg"
YOLO_WEIGHTS = "yolov3-tiny.weights"
YOLO_NAMES = "coco.names"

for f in [YOLO_CONFIG, YOLO_WEIGHTS, YOLO_NAMES]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"'{f}' topilmadi!")

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

with open(YOLO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]


# ---------------------------
# DETECTION ROUTE
# ---------------------------
@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Fayl topilmadi: 'image' kerak"}), 400

        file = request.files["image"]
        data = file.read()

        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "OpenCV decode qila olmadi"}), 400

        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        output_layers = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        results = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                results.append({
                    "label": classes[class_ids[i]],
                    "confidence": round(confidences[i] * 100)
                })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


