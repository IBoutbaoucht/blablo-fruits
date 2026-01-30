import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from src import config

class HybridAgroSystem:
    def __init__(self):
        """
        Initializes the Hybrid AI System by loading both models.
        """
        yolo_path = os.path.join(config.MODELS_DIR, config.YOLO_MODEL_NAME)
        leaf_path = os.path.join(config.MODELS_DIR, config.LEAF_MODEL_NAME)

        print(f"[INFO] Loading YOLO model from {yolo_path}...")
        self.yolo_model = YOLO(yolo_path)

        print(f"[INFO] Loading EfficientNet model from {leaf_path}...")
        self.leaf_model = tf.keras.models.load_model(leaf_path)
        
        print("[INFO] Models loaded successfully.")

    def preprocess_for_efficientnet(self, image_crop):
        """
        Resizes and normalizes a cropped image for the EfficientNet model.
        """
        try:
            # Resize to (300, 300) as per training config
            img = cv2.resize(image_crop, config.EFFICIENTNET_INPUT_SHAPE)
            
            # Convert BGR (OpenCV standard) to RGB (Keras standard)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add batch dimension (1, 300, 300, 3)
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return None

    def analyze(self, image_path):
        """
        Main pipeline:
        1. Run YOLO to find leaves/fruits.
        2. If a leaf is found, crop it and run EfficientNet for specific diagnosis.
        3. Combine results.
        """
        # Load Image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Run YOLO Inference
        yolo_results = self.yolo_model(image_path, verbose=False)
        
        final_predictions = []

        for r in yolo_results:
            for box in r.boxes:
                # Extract YOLO bounding box and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                yolo_label = config.YOLO_CLASSES[cls_id]

                # --- HYBRID LOGIC ---
                # If YOLO thinks it's a "Tomato leaf", we verify with EfficientNet
                # because EfficientNet is better at specific diseases (Bacterial vs Fungal)
                if "Tomato" in yolo_label or "leaf" in yolo_label:
                    
                    # 1. Crop the detected object
                    crop = original_img[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # 2. Preprocess for Keras
                        processed_crop = self.preprocess_for_efficientnet(crop)
                        
                        if processed_crop is not None:
                            # 3. Run EfficientNet Prediction
                            leaf_preds = self.leaf_model.predict(processed_crop, verbose=False)
                            predicted_idx = np.argmax(leaf_preds)
                            efficientnet_label = config.LEAF_DISEASE_CLASSES[predicted_idx]
                            
                            # 4. Construct Final Diagnosis
                            diagnosis = f"{efficientnet_label} (Verified)"
                        else:
                            diagnosis = yolo_label # Fallback to YOLO
                    else:
                        diagnosis = yolo_label
                else:
                    # For non-leaf items (like fruits), trust YOLO or use Fruit Model (if added)
                    diagnosis = yolo_label

                final_predictions.append({
                    "box": [x1, y1, x2, y2],
                    "label": diagnosis,
                    "confidence": conf,
                    "source": "Hybrid (YOLO + EfficientNet)"
                })

        return final_predictions
