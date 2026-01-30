import os

# --- PATHS ---
# Automatically find the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model File Names
YOLO_MODEL_NAME = "best_yolo.pt"
LEAF_MODEL_NAME = "leaf_model.keras"

# --- MODEL PARAMETERS ---
# Verified from Diagnostic Scan
EFFICIENTNET_INPUT_SHAPE = (300, 300) 

# --- CLASS MAPPINGS ---

# 1. EFFICIENTNET CLASSES (The 8 classes we found)
LEAF_DISEASE_CLASSES = [
    'bacterial_disease', 
    'fungal_disease', 
    'healthy', 
    'insect_damage', 
    'magnesium_deficiency', 
    'nitrogen_deficiency', 
    'potassium_deficiency', 
    'viral_disease'
]

# 2. FRUIT CLASSES (Ripe/Unripe/Spoiled)
FRUIT_STATUS_CLASSES = ['Ripe', 'Spoiled', 'Unripe']

# 3. YOLO CLASSES (PlantDoc Standard - 30 classes)
# Used to determine if a detection is a "Leaf" or a "Fruit"
YOLO_CLASSES = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf spot',
    'Bell_pepper leaf', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot',
    'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight',
    'Potato leaf late blight', 'Potato leaf', 'Raspberry leaf', 'Soyabean leaf',
    'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf bacterial spot',
    'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus',
    'Tomato leaf', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf',
    'grape leaf black rot', 'grape leaf'
]
