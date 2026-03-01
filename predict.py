import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --------- SET IMAGE PATH HERE ---------
IMAGE_PATH = "test.jpg"
# ---------------------------------------

# Load model
model = tf.keras.models.load_model("plant_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# Load and preprocess image
image = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)
class_index = np.argmax(prediction)
confidence = float(np.max(prediction))

print("Predicted Disease:", class_names[class_index])
print("Confidence:", round(confidence * 100, 2), "%")