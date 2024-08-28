#######The code that i have commented, is for locahost execution and if you have to publish then there are some minor changes ...for better understanding i have kept both the code 


# from flask import Flask, json, request
# from tensorflow.keras.models import model_from_json
# from flask_cors import CORS
# import numpy as np
# import cv2
# import base64
# from io import BytesIO
# from PIL import Image

# app = Flask(__name__)
# CORS(app)

# # Load model
# # Opens the model.json file containing the structure of the pre-trained Keras model and loads it.
# # The model_from_json function creates a Keras model from the loaded JSON.
# # The model weights are loaded from the model.h5 file, which contains the trained weights for this model.

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")


# #This function converts a base64-encoded image string into an OpenCV cv2 image.
# # b64str.split(',')[1]: Splits the base64 string to remove metadata (data:image/jpeg;base64,).
# # np.frombuffer: Decodes the base64 string into a NumPy array.
# # cv2.imdecode: Converts the NumPy array into an image using OpenCV.

# def get_cv2_image_from_base64_string(b64str):
#     encoded_data = b64str.split(',')[1]
#     nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

# #Defines a simple GET request at the root ("/") that returns a welcome message
# @app.route("/", methods=['GET'])
# def get_home():
#     return "Welcome to the Brain Tumor Detection API!"


# # This POST route processes the incoming image(s) and makes predictions.
# # json.loads(request.data): Parses the incoming JSON request data.
# # Iterates over the images provided in the data['image'] field (assumed to be base64 strings), decodes each image using get_cv2_image_from_base64_string, and resizes it to 224x224 pixels (the input size for the pre-trained model).
# # Collects the processed images into a list (predict_img), which is then converted to a NumPy array and passed to the model for prediction (loaded_model.predict).
# # The predictions are post-processed to extract the class with the highest probability (np.argmax(prediction, axis=1)), but the API returns the probability of class 1 (prediction[:, 1]).
# @app.route("/", methods=['POST'])
# def read_root():
#     data = json.loads(request.data)
#     predict_img = []
#     for item in data['image']:
#         image = get_cv2_image_from_base64_string(item)
#         image = cv2.resize(image, (224, 224))
#         predict_img.append(image)

#     prediction = loaded_model.predict(np.array(predict_img))
#     result = np.argmax(prediction, axis=1)

#     return {"result": prediction[:, 1].tolist()}

# # Handle favicon requests to avoid 404 errors
# @app.route('/favicon.ico')
# def favicon():
#     return '', 204

# if __name__ == '__main__':
#     app.run(port=5000)


# ####Summary####
# # This Flask API accepts a POST request with base64-encoded images of brain scans, processes them, and returns a prediction from a pre-trained model.
# # It allows communication between the front end (like a React app) and the machine learning model, supporting cross-origin requests with CORS.

from flask import Flask, json, request
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load model
loaded_model = load_model('model.h5')  # Load the model saved with `model.save('model')`

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route("/", methods=['GET'])
def get_home():
    return "Welcome to the Brain Tumor Detection API!"

@app.route("/", methods=['POST'])
def read_root():
    data = json.loads(request.data)
    predict_img = []
    for item in data['image']:
        image = get_cv2_image_from_base64_string(item)
        image = cv2.resize(image, (224, 224))
        predict_img.append(image)

    prediction = loaded_model.predict(np.array(predict_img))
    result = np.argmax(prediction, axis=1)

    return {"result": prediction[:, 1].tolist()}

# Handle favicon requests to avoid 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
