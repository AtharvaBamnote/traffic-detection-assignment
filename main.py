from detector import detect_vehicles
from utils import download_image
import os
import cv2
import supervision as sv

if __name__ == "__main__":
    
    os.makedirs("output/processed_images", exist_ok=True)   
    image_url = "https://images.hindustantimes.com/auto/img/2022/09/07/1600x900/Mumbai_Pune_Expressway_1662525979768_1662525979913_1662525979913.jpg"
    image = download_image(image_url)    
    annotated_image, vehicle_counts = detect_vehicles(image)
    save_path = "output/processed_images/annotated_image.jpg"
    cv2.imwrite(save_path, annotated_image)
    sv.plot_image(annotated_image)
    print(f"✅ Saved to: {save_path}")
    print(f"📊 Detected vehicles: {dict(vehicle_counts)}")
