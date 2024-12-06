from flask import Flask, request, jsonify
import json
import os
from .model import VegeCareModel

app = Flask(__name__)
model = VegeCareModel("/app/models/vegecare_model_mobilenetv2finetune.h5")

# Load care recommendations
with open("/app/data/care_recommendations.json", "r") as f:
    CARE_RECOMMENDATIONS = json.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    # Predict disease
    prediction = model.predict(image_bytes)

    # Get care recommendations
    care_info = CARE_RECOMMENDATIONS.get(
        prediction["class"],
        {
            "general_care": "Consult a local agricultural expert for specific care instructions.",
            "treatment": "No specific treatment found.",
            "prevention": "Maintain good plant hygiene and monitoring.",
        },
    )

    response = {"disease_prediction": prediction, "plant_care": care_info}

    return jsonify(response)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200
