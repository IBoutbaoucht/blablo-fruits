import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image

class HybridAgroModel:
    def __init__(self, yolo_path, leaf_keras_path, fruit_keras_path=None):
        """
        Initializes the hybrid system.
        """
        print("Loading YOLOv8 (Detection + Classification)...")
        self.yolo = YOLO(yolo_path)
        
        print("Loading EfficientNet (Leaf Verification)...")
        self.leaf_model = tf.keras.models.load_model(leaf_keras_path)
        
        # Optional: Load fruit model if path provided
        if fruit_keras_path:
            print("Loading Fruit Classifier...")
            self.fruit_model = tf.keras.models.load_model(fruit_keras_path)
            self.fruit_classes = ['Ripe', 'Spoiled', 'Unripe'] # Hardcoded from your prompt
        else:
            self.fruit_model = None

    def preprocess_for_keras(self, crop_img, target_size=(224, 224)):
        """
        Prepares a YOLO crop for EfficientNet.
        """
        # Resize to EfficientNet's expected input
        img = cv2.resize(crop_img, target_size)
        # Convert BGR to RGB (OpenCV uses BGR, Keras expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize (Check your training notebook: did you use 1/255 or -1 to 1?)
        # Assuming 1/255 for standard EfficientNet transfer learning
        img = img.astype('float32') / 255.0
        # Add batch dimension (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        return img

    def analyze_image(self, image_path):
        # 1. Run YOLO Inference
        yolo_results = self.yolo(image_path)
        
        # Load original image for cropping
        original_img = cv2.imread(image_path)
        
        final_report = []

        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                # Get YOLO data
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                yolo_label = self.yolo.names[cls_id]
                
                # Crop the object
                crop = original_img[y1:y2, x1:x2]
                
                # Decision Logic
                if "Leaf" in yolo_label or "leaf" in yolo_label:
                    # Double check with EfficientNet
                    processed_crop = self.preprocess_for_keras(crop)
                    leaf_preds = self.leaf_model.predict(processed_crop, verbose=0)
                    leaf_class_id = np.argmax(leaf_preds)
                    # Note: You need the class list from Script 1 to map this ID to a name
                    final_label = f"YOLO says {yolo_label} | EfficientNet ID {leaf_class_id}"
                    
                elif "Fruit" in yolo_label or "Tomato" in yolo_label:
                    if self.fruit_model:
                        processed_crop = self.preprocess_for_keras(crop)
                        fruit_preds = self.fruit_model.predict(processed_crop, verbose=0)
                        fruit_class = self.fruit_classes[np.argmax(fruit_preds)]
                        final_label = f"Tomato: {fruit_class}"
                    else:
                        final_label = yolo_label
                else:
                    final_label = yolo_label

                final_report.append({
                    "bbox": [x1, y1, x2, y2],
                    "prediction": final_label,
                    "confidence": conf
                })
        
        return final_report

# Usage Example (Commented out for script file)
# system = HybridAgroModel('path/to/yolo.pt', 'path/to/leaf.keras', 'path/to/fruit.keras')
# results = system.analyze_image('test_plant.jpg')
# print(results)
