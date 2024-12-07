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
        return jsonify({"error": "Unsupported file type. Please upload a PNG, JPG, or JPEG file."}), 400

    image_bytes = image_file.read()

    prediction = model.predict(image_bytes)

    care_info = CARE_RECOMMENDATIONS.get(
        prediction["condition"],
        {
            "general_info": "This is some general information about the plant disease.",
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
