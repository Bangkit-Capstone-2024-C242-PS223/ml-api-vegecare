from flask import Flask, request, jsonify
import json
import os
from .model import VegeCareModel

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app = Flask(__name__)
model = VegeCareModel("/app/models/vegecare_model_mobilenetv2finetune.h5")

# Load care recommendations
with open("/app/data/care_recommendations_bahasa.json", "r") as f:
    CARE_RECOMMENDATIONS = json.load(f)


def allowed_file(filename):
    # Memastikan file memiliki ekstensi yang valid
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    if not allowed_file(image_file.filename):
        return (
            jsonify(
                {
                    "error": "Unsupported file type. Please upload a PNG, JPG, or JPEG file."
                }
            ),
            400,
        )

    image_bytes = image_file.read()

    prediction = model.predict(image_bytes)

    full_key = f"{prediction['plant_name']}__{prediction['condition']}"

    care_info = CARE_RECOMMENDATIONS.get(
        full_key,
        {
            "general_info": "No specific information available.",
            "general_care": "Maintain regular plant care practices.",
            "treatment": "No special treatment required.",
            "prevention": "Monitor plant health regularly.",
        },
    )

    response = {
        "prediction": {
            "plant_name": prediction["plant_name"],
            "condition": prediction["condition"],
            "confidence": prediction["confidence"],
        },
        "plant_care": {
            "general_info": care_info.get("general_info"),
            "general_care": care_info.get("general_care"),
            "treatment": care_info.get("treatment"),
            "prevention": care_info.get("prevention"),
        },
    }

    return jsonify(response)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200
