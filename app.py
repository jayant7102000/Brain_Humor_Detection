######The code that i have commented, is for locahost execution and if you have to publish then there are some minor changes ...for better understanding i have kept both the code 


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





# # use fast api taking string as parameter from a get function and converting it to image and then to numpy array and then to a dataframe and then to a prediction and then to a json file and then to a string and then to a response
# from flask import Flask, json, request
# from tensorflow.keras.models import model_from_json
# from flask_cors import CORS, cross_origin
# import numpy as np
# import pandas as pd
# import cv2
# import pickle
# import base64
# from io import BytesIO
# from PIL import Image
# from typing import List
# from pydantic import BaseModel
# import tensorflow as tf

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# # model = pickle.load(open("brain_tumor_model.pkl", "rb"))
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")



# def get_cv2_image_from_base64_string(b64str):
#     encoded_data = b64str.split(',')[1]
#     nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img


# def get_image_from_base64_string(b64str):
#     encoded_data = b64str.split(',')[1]
#     image_data = BytesIO(base64.b64decode(encoded_data))
#     img = Image.open(image_data)
#     return img

# @app.route('/',methods=['GET'])
# def home():
#     return "Hello World"


# @app.route("/", methods=['POST'])
# def read_root():
#     data = json.loads(request.data)
#     predict_img = []
#     for item in data['image']:
#         #Decode the base64-encoded image
#         image = get_cv2_image_from_base64_string(item)
#         image = cv2.resize(image,(224,224))
#         predict_img.append(image)
#         # encoded_data = item.split(',')[1]
#         # image_data = BytesIO(base64.b64decode(encoded_data))
#         # pil_image = Image.open(image_data)
#         # # Resize the image to 224x224
#         # resized_image = pil_image.resize((224, 224))
#         # # Append the resized image to the list
#         # predict_img.append(resized_image)

#     # np_images = np.array([np.array(img) for img in predict_img])
#     # # Convert the NumPy array to a TensorFlow tensor
#     # tf_images = tf.convert_to_tensor(np_images, dtype=tf.float32)
#     # # # Convert the image to a numpy array
#     prediction = loaded_model.predict(np.array(predict_img))
#     result = np.argmax(prediction, axis=1)

#     # make the probablity frtom prediction
#     # print(prediction[:,1])
#     # print(result)

#     return {"result": prediction[:, 1].tolist()}


# if __name__ == '__main__':
#     app.run(port=5000)



from flask import Flask, json, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
# Configure CORS to allow requests from your Netlify site
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the model
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/', methods=['GET'])
def home():
    return "Hello World"

@app.route("/", methods=['POST'])
def read_root():
    try:
        data = json.loads(request.data)
        images = data.get('image', [])
        
        if not images:
            return jsonify({"error": "No images provided"}), 400

        predict_img = []
        for item in images:
            image = get_cv2_image_from_base64_string(item)
            image = cv2.resize(image, (224, 224))
            predict_img.append(image)

        np_images = np.array(predict_img)
        predictions = loaded_model.predict(np_images)
        result = np.argmax(predictions, axis=1)
        probabilities = predictions[:, 1].tolist()

        return jsonify({"result": probabilities})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
