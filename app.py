# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json
# import io

# # -----------------------------
# # Load ML Model (Only Once)
# # -----------------------------
# model = tf.keras.models.load_model("plant_model.keras")

# with open("class_names.json", "r") as f:
#     class_names_raw = json.load(f)

# # Reverse mapping: index -> class name
# index_to_class = {v: k for k, v in class_names_raw.items()}

# # -----------------------------
# # Initialize FastAPI
# # -----------------------------
# app = FastAPI()

# # Enable CORS (for frontend like Netlify)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to frontend URL after deployment
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # Store Latest Sensor Data
# # -----------------------------
# latest_sensor_data = {
#     "temperature": None,
#     "humidity": None,
#     "moisture": None,
#     "pump": "OFF",
#     "mode": "AUTO"
# }

# # -----------------------------
# # Sensor Data Model
# # -----------------------------
# class SensorData(BaseModel):
#     temperature: float
#     humidity: float
#     moisture: int
#     pump: str

# # -----------------------------
# # Root Endpoint
# # -----------------------------
# @app.get("/")
# def home():
#     return {"message": "Plant Disease Detection API Running 🌱"}

# # -----------------------------
# # Disease Prediction Endpoint
# # -----------------------------
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()

#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         image = image.resize((224, 224))

#         image = np.array(image).astype("float32") / 255.0
#         image = np.expand_dims(image, axis=0)

#         predictions = model.predict(image)

#         predicted_class = int(np.argmax(predictions[0]))
#         confidence = float(np.max(predictions[0]) * 100)

#         disease_name = index_to_class[predicted_class]

#         return {
#             "disease": disease_name,
#             "confidence": round(confidence, 2)
#         }

#     except Exception as e:
#         return {"error": str(e)}

# # -----------------------------
# # Receive Sensor Data (ESP32)
# # -----------------------------
# @app.post("/sensor-data")
# def receive_sensor_data(data: SensorData):
#     global latest_sensor_data

#     latest_sensor_data["temperature"] = data.temperature
#     latest_sensor_data["humidity"] = data.humidity
#     latest_sensor_data["moisture"] = data.moisture

#     # AUTO mode irrigation logic
#     if latest_sensor_data["mode"] == "AUTO":
#         if data.moisture < 1500:
#             latest_sensor_data["pump"] = "ON"
#         else:
#             latest_sensor_data["pump"] = "OFF"
#     else:
#         latest_sensor_data["pump"] = data.pump

#     return {
#         "message": "Sensor data received",
#         "pump": latest_sensor_data["pump"]
#     }

# # -----------------------------
# # Get Latest Sensor Data
# # -----------------------------
# @app.get("/sensor-data")
# def get_sensor_data():
#     return latest_sensor_data

# # -----------------------------
# # Pump Control
# # -----------------------------
# @app.post("/pump-control")
# def control_pump(mode: str):
#     global latest_sensor_data

#     if mode.upper() == "AUTO":
#         latest_sensor_data["mode"] = "AUTO"

#     elif mode.upper() == "ON":
#         latest_sensor_data["mode"] = "MANUAL"
#         latest_sensor_data["pump"] = "ON"

#     elif mode.upper() == "OFF":
#         latest_sensor_data["mode"] = "MANUAL"
#         latest_sensor_data["pump"] = "OFF"

#     return {
#         "mode": latest_sensor_data["mode"],
#         "pump": latest_sensor_data["pump"]
#     }
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io

# -----------------------------
# Load ML Model (Only Once)
# -----------------------------
model = tf.keras.models.load_model("plant_model.keras")

with open("class_names.json", "r") as f:
    class_names_raw = json.load(f)

index_to_class = {v: k for k, v in class_names_raw.items()}

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GLOBAL SYSTEM STATE
# -----------------------------
pump_mode = "AUTO"   # AUTO / ON / OFF
pump_state = "OFF"

latest_sensor_data = {
    "temperature": 0,
    "humidity": 0,
    "moisture": 0
}

# -----------------------------
# Models
# -----------------------------
class SensorData(BaseModel):
    temperature: float
    humidity: float
    moisture: int

class PumpControl(BaseModel):
    mode: str   # AUTO / ON / OFF


# -----------------------------
# Root
# -----------------------------
@app.get("/")
def home():
    return {"message": "Plant Disease + Smart Irrigation API Running 🌱"}


# -----------------------------
# Disease Prediction
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        image = np.array(image).astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)

        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)

        disease_name = index_to_class[predicted_class]

        return {
            "disease": disease_name,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Receive Sensor Data (ESP32)
# -----------------------------
@app.post("/sensor-data")
def receive_sensor_data(data: SensorData):
    global latest_sensor_data, pump_mode, pump_state

    latest_sensor_data = data.dict()
    moisture = latest_sensor_data["moisture"]

    # -------- Pump Logic --------
    if pump_mode == "ON":
        pump_state = "ON"

    elif pump_mode == "OFF":
        pump_state = "OFF"

    else:  # AUTO mode
        if moisture < 30:   # because ESP32 sends 0-100 %
            pump_state = "ON"
        else:
            pump_state = "OFF"

    return {
        "pump_mode": pump_mode,
        "pump_state": pump_state
    }


# -----------------------------
# Get Sensor Data (Frontend)
# -----------------------------
@app.get("/sensor-data")
def get_sensor_data():
    return {
        "temperature": latest_sensor_data["temperature"],
        "humidity": latest_sensor_data["humidity"],
        "moisture": latest_sensor_data["moisture"],
        "pump_mode": pump_mode,
        "pump_state": pump_state
    }


# -----------------------------
# Pump Control (Frontend)
# -----------------------------
@app.post("/pump-control")
def control_pump(control: PumpControl):
    global pump_mode

    mode = control.mode.upper()

    if mode in ["AUTO", "ON", "OFF"]:
        pump_mode = mode
        return {
            "message": f"Pump mode set to {pump_mode}"
        }

    return {"error": "Invalid mode. Use AUTO / ON / OFF"}
import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)