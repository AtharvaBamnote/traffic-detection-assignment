import supervision as sv
from inference import get_model
from collections import Counter
import cv2
import numpy as np

vehicle_classes = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

color_map = {
    'bicycle': (255, 0, 255),
    'car': (0, 255, 0),
    'motorcycle': (255, 0, 0),
    'bus': (0, 0, 255),
    'truck': (0, 255, 255)
}

def detect_vehicles(image):
    model = get_model(model_id="yolov8l-640")
    detections_list = []
    scales = [1.0, 0.75, 0.5]

    for scale in scales:
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        results = model.infer(scaled, confidence=0.2)[0]
        det = sv.Detections.from_inference(results)

        if len(det.xyxy) > 0:
            det.xyxy = det.xyxy / scale
            detections_list.append(det)

    if detections_list:
        detections = sv.Detections.merge(detections_list).with_nms(threshold=0.6)
    else:
        detections = sv.Detections(xyxy=np.array([]), confidence=np.array([]), class_id=np.array([]))

    filtered_indices = []
    for i, (xyxy, cls_id, conf) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
        if cls_id in vehicle_classes and conf >= 0.2:
            x1, y1, x2, y2 = xyxy
            area = (x2 - x1) * (y2 - y1)
            if area >= 30:
                filtered_indices.append(i)

    if filtered_indices:
        detections = sv.Detections(
            xyxy=detections.xyxy[filtered_indices],
            confidence=detections.confidence[filtered_indices],
            class_id=detections.class_id[filtered_indices],
            tracker_id=None,
            data=None
        )
    else:
        detections = sv.Detections(xyxy=np.array([]), confidence=np.array([]), class_id=np.array([]))

    vehicle_counts = Counter()
    for cls_id in detections.class_id:
        vehicle_counts[vehicle_classes[cls_id]] += 1

    labels = [f"{vehicle_classes[cls]} {conf:.2f}" for cls, conf in zip(detections.class_id, detections.confidence)]
    colors = [color_map[vehicle_classes[cls]] for cls in detections.class_id]

    annotated = image.copy()
    for box, label, color in zip(detections.xyxy, labels, colors):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    summary_text = f"Vehicle Counts: {dict(vehicle_counts)}"
    font_scale = 0.8
    text_size, _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, annotated.shape[0] - text_size[1] - 20), (10 + text_size[0] + 10, annotated.shape[0] - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
    cv2.putText(annotated, summary_text, (15, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return annotated, vehicle_counts
