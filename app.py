from joblib import load
import cv2
from flask import Flask, request, jsonify



app = Flask(__name__)

@app.route('/')
def myroute():
    return "hellow marwan"

@app.route('/prediction', methods=["POST"])
def index():
    # loading the model
    loaded_model = load("brain_tumot_svm.h5")

    #loading new image
    image = request.files['image']
    image.save('img.jpg')

    # new image processing
    image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (240, 240))
    image_flat = image_resized.reshape(1, -1) / 255.0  # Flatten and normalize

    # Ensure the shape matches the model's expectation
    if image_flat.shape[1] != 57600:
        return jsonify({"error": "Invalid image dimensions"}), 400

    result = loaded_model.predict(image_flat)
    if result == 0:
        results = {"diag": "no tumor"}
    elif result == 1:
        results = {"diag": "tumor"}
    else:
        results = {"diag": "none"}

    return jsonify(results)



if __name__ == "__main__":
    app.run('0.0.0.0',9090)