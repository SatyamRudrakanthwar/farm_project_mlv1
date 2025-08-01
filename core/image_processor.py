import cv2
import supervision as sv
from ultralytics import YOLO

# Load the trained YOLO model
model_path = "D:/projects/AgriSavant/ETL PEST DETECTION CODE MODULARITY/pest_etl_app/models/best.pt"
model = YOLO(model_path)

def process_image(image_path):
    try:
        # Perform inference
        results = model(image_path)
        detections = sv.Detections.from_ultralytics(results[0])

        # Get class names from detections
        class_names = [model.names[int(cls)] for cls in detections.class_id]
        pest_counts_local = {name: class_names.count(name) for name in set(class_names)}

        # Load image and annotate it
        image = cv2.imread(image_path)
        if image is None:
            return None, {}, "Error reading image"

        # Draw annotations
        annotated = sv.BoxAnnotator().annotate(scene=image, detections=detections)
        annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

        return annotated, pest_counts_local, None

    except Exception as e:
        return None, {}, str(e)
