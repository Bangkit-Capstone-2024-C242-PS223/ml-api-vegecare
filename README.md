# Machine Learning API

## Tools
- [Google Cloud Platform](https://cloud.google.com/)
- [Flask](https://flask.palletsprojects.com/)
- [Postman](https://www.postman.com/)

## Prerequisites
- Python 3.8+
- TensorFlow
- Flask
- Pillow
- NumPy

## Installation
1. Clone the repository
```bash
git clone
```
2. Install the required packages
```bash
pip install -r requirements.txt
```
3. Run the Flask server
```bash
python app.py
```
4. Open Postman and make a POST request to `http://localhost:5000/predict` with the image file as form-data

Output example:
```
{
    "plant_care": {
        "general_care": "Hindari penanganan tanaman saat basah dan pastikan jarak tanam yang tepat untuk meningkatkan sirkulasi udara",
        "general_info": "Penyakit ini disebabkan oleh bakteri yang menyebabkan bercak-bercak kecil coklat hingga hitam pada daun dan batang yang dapat membesar dan menyebabkan jaringan tanaman membusuk",
        "prevention": "Gunakan benih yang bebas penyakit dan lakukan rotasi tanaman dengan tanaman non-kubis untuk mencegah akumulasi patogen di tanah",
        "treatment": "Buang dan musnahkan bagian tanaman yang terinfeksi. Aplikasikan bakterisida berbahan dasar tembaga sesuai petunjuk"
    },
    "prediction": {
        "condition": "Bercak Bakteri",
        "confidence": 0.9999657869338989,
        "plant_name": "Kembang Kol"
    }
}
```

## Docker Deployment
```bash
bashCopydocker build -t vegecare-api .
docker run -p 5000:5000 vegecare-api
```
