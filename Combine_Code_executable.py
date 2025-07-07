import supervision as sv
from inference import get_model
from inference.core.utils.image_utils import load_image_bgr
import cv2
from collections import Counter
import numpy as np

# Load image
try:
    image = load_image_bgr("") #coopy path of the image 
    if image is None:
        raise ValueError("Failed to load image from URL")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Load YOLOv8 large model
model = get_model(model_id="yolov8l-640")

# Multi-scale inference to detect vehicles of varying sizes
detections_list = []
scales = [1.0, 0.75, 0.5] 
for scale in scales:
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        results = model.infer(scaled_image, confidence=0.2)[0]  
        scaled_detections = sv.Detections.from_inference(results)
        if len(scaled_detections.xyxy) > 0:
            scaled_detections.xyxy = scaled_detections.xyxy / scale
            detections_list.append(scaled_detections)
    except Exception as e:
        print(f"Error during inference at scale {scale}: {e}")
        continue
if detections_list:
    detections = sv.Detections.merge(detections_list)
else:
    print("No detections from any scale")
    detections = sv.Detections(xyxy=np.array([]), confidence=np.array([]), class_id=np.array([]))
detections = detections.with_nms(threshold=0.6)

# Define expanded vehicle class mappings (COCO dataset)
vehicle_classes = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

min_box_area = 30
filtered_indices = []
for i, (xyxy, cls_id, conf) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
    if cls_id in vehicle_classes and conf >= 0.2:
        x1, y1, x2, y2 = xyxy
        area = (x2 - x1) * (y2 - y1)
        if area >= min_box_area:
            filtered_indices.append(i)

# Apply filters
if filtered_indices:
    detections = sv.Detections(
        xyxy=detections.xyxy[filtered_indices],
        confidence=detections.confidence[filtered_indices],
        class_id=detections.class_id[filtered_indices],
        tracker_id=None if detections.tracker_id is None else detections.tracker_id[filtered_indices],
        data={k: v[filtered_indices] for k, v in detections.data.items()} if detections.data else None
    )
else:
    print("No vehicle detections after filtering")
    detections = sv.Detections(xyxy=np.array([]), confidence=np.array([]), class_id=np.array([]))

# Count vehicles by type
vehicle_counts = Counter()
for cls_id in detections.class_id:
    vehicle_counts[vehicle_classes[cls_id]] += 1

# Define color mapping for each vehicle type (BGR format)
color_map = {
    'bicycle': (255, 0, 255),   # Magenta
    'car': (0, 255, 0),         # Green
    'motorcycle': (255, 0, 0),  # Blue
    'bus': (0, 0, 255),         # Red
    'truck': (0, 255, 255),    # Yellow
}

# Annotate image with bounding boxes and custom colors
annotated_image = image.copy()
labels = [f"{vehicle_classes[cls_id]} {conf:.2f}" for cls_id, conf in zip(detections.class_id, detections.confidence)]
colors = [color_map[vehicle_classes[cls_id]] for cls_id in detections.class_id]

# Custom annotation with colored boxes and labels
for box, label, color in zip(detections.xyxy, labels, colors):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        annotated_image, 
        label, 
        (x1, y1 - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        color, 
        2, 
        cv2.LINE_AA
    )

# Add summary text 
summary_text = f"Vehicle Counts: {dict(vehicle_counts)}"
font_scale = 0.8
text_size, _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
text_w, text_h = text_size
overlay = annotated_image.copy()
cv2.rectangle(
    overlay,
    (10, annotated_image.shape[0] - text_h - 20),
    (10 + text_w + 10, annotated_image.shape[0] - 10),
    (0, 0, 0, 180),
    -1
)
alpha = 0.5
cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
cv2.putText(
    img=annotated_image,
    text=summary_text,
    org=(15, annotated_image.shape[0] - 15),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=font_scale,
    color=(255, 255, 255),
    thickness=2,
    lineType=cv2.LINE_AA
)
try:
    cv2.imwrite("annotated_vehicles_fullscan.jpg", annotated_image)
    sv.plot_image(annotated_image)
    print(f"Annotated image saved as 'annotated_vehicles_fullscan.jpg'")
    print(f"Detected vehicles: {dict(vehicle_counts)}")
except Exception as e:
    print(f"Error saving or displaying image: {e}")
