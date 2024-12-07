import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io


class VegeCareModel:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.classes = [
            "Cabai__Healthy",
            "Cabai__Leaf_Curl",
            "Cabai__Leaf_Spot",
            "Cabai__Whitefly",
            "Cabai__Yellowish",
            "Kembang Kol__Bacterial_Spot_Rot",
            "Kembang Kol__Black_Rot",
            "Kembang Kol__Downy_Mildew",
            "Kembang Kol__Healthy",
            "Lettuce_Bacterial",
            "Lettuce__Fungal",
            "Lettuce__Healthy",
            "Sawi_Hama_Ulat_Grayak",
            "Sawi_Healthy",
            "Terong__Healthy_Leaf",
            "Terong__Insect_Pest_Disease",
            "Terong__Leaf_Spot_Disease",
            "Terong__Mosaic_Virus_Disease",
            "Terong__Small_Leaf_Disease",
            "Terong__White_Mold_Disease",
            "Terong__Wilt_Disease",
            "Timun__Anthracnose",
            "Timun__Bacterial_Wilt",
            "Timun__Belly_Rot",
            "Timun__Downy_Mildew",
            "Timun__Fresh_Leaf",
            "Timun__Gummy_Stem_Blight",
            "Timun__Pythium_Fruit_Rot",
            "Tomato__Bacterial_Spot",
            "Tomato__Early_Blight",
            "Tomato__Healthy",
            "Tomato__Late_Blight",
            "Tomato__Leaf_Mold",
            "Tomato__Septoria_Leaf_Spot",
            "Tomato__Spider_Mites Two-Spotted_Spider_Mite",
            "Tomato__Target_Spot",
            "Tomato__Tomato_Mosaic_Virus",
            "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
        ]

    def parse_prediction(self, predicted_class):
        if "__" in predicted_class:
            plant_name, condition = predicted_class.split("__")
        else:
            plant_name, condition = predicted_class, "Unknown"
        return plant_name, condition


    def preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image_bytes):
        processed_image = self.preprocess_image(image_bytes)
        predictions = self.model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = self.classes[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        plant_name, condition = self.parse_prediction(predicted_class)

        return {
            "plant_name": plant_name,
            "condition": condition,
            "confidence": confidence
        }

