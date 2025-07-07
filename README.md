# Traffic-detection-assignment
With YOLOv8 and Supervision
This project uses YOLOv8 and the supervision library to detect and classify vehicles in an image. It processes the image at multiple scales to ensure comprehensive detection across the entire image, identifying vehicle types such as bicycles, cars, motorcycles, buses, and trucks. The results are visualized with color-coded bounding boxes and a summary of vehicle counts.
Features

Multi-Scale Detection: Processes the image at different resolutions to detect vehicles of varying sizes.
Vehicle Classification: Identifies bicycles, cars, motorcycles, buses, and trucks using the COCO dataset class mappings.
Visualization: Draws color-coded bounding boxes and labels for each detected vehicle.
Output: Saves the annotated image and prints a summary of detected vehicle counts.

Requirements

Python 3.8+
Libraries:
supervision (pip install supervision)
inference (pip install inference)
opencv-python (pip install opencv-python)
numpy (pip install numpy)



Installation

Clone or download this repository.
Install the required libraries:pip install supervision inference opencv-python numpy


Ensure you have a working internet connection to download the YOLOv8 model weights during the first run.

Usage

Place the vehicle_detection.py script in your working directory.
Run the script:python vehicle_detection.py


The script will:
Load an image from the specified URL (Mumbai-Pune Expressway image by default).
Perform vehicle detection using the YOLOv8 large model (yolov8l-640).
Save the annotated image as annotated_vehicles_fullscan.jpg.
Display the annotated image and print vehicle counts to the console.



Configuration

Image Input: Modify the load_image_bgr call in vehicle_detection.py to use a local image or video:image = load_image_bgr("path/to/your/image.jpg")


Model: Replace yolov8l-640 with yolov8n-640 for faster inference or yolov8x-640 for higher accuracy.
Parameters:
confidence=0.2: Adjusts detection sensitivity (lower values detect more objects but may include false positives).
min_box_area=30: Minimum bounding box area to filter out tiny detections.
scales=[1.0, 0.75, 0.5]: Adjust scales for multi-scale detection (fewer scales for faster processing).



Output

Annotated Image: Saved as annotated_vehicles_fullscan.jpg with color-coded bounding boxes (e.g., green for cars, blue for motorcycles).
Console Output: Displays a dictionary of vehicle counts (e.g., {'car': 10, 'motorcycle': 3, 'bus': 1}).

Notes

Performance: Multi-scale inference increases accuracy but is computationally intensive. Use a GPU with CUDA for faster processing.
Limitations: The code uses COCO dataset classes, which may not detect specialized vehicle types (e.g., SUVs vs. sedans). For custom vehicle types, train a YOLO model on a specific dataset.
Image Context: The default image is a highway scene, likely containing cars, motorcycles, buses, and trucks. Bicycles, trains, boats, or airplanes may require different images for testing.
Enhancements:
Add video support by replacing load_image_bgr with cv2.VideoCapture.
Integrate tracking (e.g., DeepSORT) for crowded scenes or video.
Apply image preprocessing (e.g., contrast enhancement) for better detection of distant vehicles.



Troubleshooting

Image Loading Error: Ensure the URL is accessible or use a local image.
No Detections: Adjust confidence (lower) or min_box_area (lower) to increase sensitivity.
Performance Issues: Reduce scales or use a lighter model (yolov8n-640).

License
This project is licensed under the MIT License.
