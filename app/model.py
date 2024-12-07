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

        self.translation_map = {
            "Cabai__Healthy": "Cabai__Sehat",
            "Cabai__Leaf_Curl": "Cabai__Daun Keriting",
            "Cabai__Leaf_Spot": "Cabai__Bercak Daun",
            "Cabai__Whitefly": "Cabai__Lalat Putih",
            "Cabai__Yellowish": "Cabai__Kuning",
            "Kembang Kol__Bacterial_Spot_Rot": "Kembang Kol__Bercak Bakteri",
            "Kembang Kol__Black_Rot": "Kembang Kol__Busuk Hitam",
            "Kembang Kol__Downy_Mildew": "Kembang Kol__Jamur Berembun",
            "Kembang Kol__Healthy": "Kembang Kol__Sehat",
            "Lettuce_Bacterial": "Selada__Bakteri",
            "Lettuce__Fungal": "Selada__Jamur",
            "Lettuce__Healthy": "Selada__Sehat",
            "Sawi_Hama_Ulat_Grayak": "Sawi__Hama Ulat Grayak",
            "Sawi_Healthy": "Sawi__Sehat",
            "Terong__Healthy_Leaf": "Terong__Sehat",
            "Terong__Insect_Pest_Disease": "Terong__Hama Kutu Daun",
            "Terong__Leaf_Spot_Disease": "Terong__Bercak Daun",
            "Terong__Mosaic_Virus_Disease": "Terong__Virus Mosaik",
            "Terong__Small_Leaf_Disease": "Terong__Daun Kecil",
            "Terong__White_Mold_Disease": "Terong__Jamur Putih",
            "Terong__Wilt_Disease": "Terong__Layu",
            "Timun__Anthracnose": "Timun__Antraknosa",
            "Timun__Bacterial_Wilt": "Timun__Layu Bakteri",
            "Timun__Belly_Rot": "Timun__Perut Busuk",
            "Timun__Downy_Mildew": "Timun__Jamur Berembun",
            "Timun__Fresh_Leaf": "Timun__Sehat",
            "Timun__Gummy_Stem_Blight": "Timun__Batang Busuk Bergetah",
            "Timun__Pythium_Fruit_Rot": "Timun__Buah Busuk Pythium",
            "Tomato__Bacterial_Spot": "Tomat__Bercak Bakteri",
            "Tomato__Early_Blight": "Tomat__Hawar Daun",
            "Tomato__Healthy": "Tomat__Sehat",
            "Tomato__Late_Blight": "Tomat__Hawar Lanjut",
            "Tomato__Leaf_Mold": "Tomat__Jamur Daun",
            "Tomato__Septoria_Leaf_Spot": "Tomat__Bercak Daun Septoria",
            "Tomato__Spider_Mites Two-Spotted_Spider_Mite": "Tomat__Kutu Laba-laba",
            "Tomato__Target_Spot": "Tomat__Bercak Target",
            "Tomato__Tomato_Mosaic_Virus": "Tomat__Virus Mosaik",
            "Tomato__Tomato_Yellow_Leaf_Curl_Virus": "Tomat__Kuning Keriting",
        }

    def translate_label(self, label):
        return self.translation_map.get(label, label)

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

        translated_class = self.translate_label(predicted_class)
        plant_name, condition = self.parse_prediction(translated_class)

        return {
            "plant_name": plant_name,
            "condition": condition,
            "confidence": confidence
        }
