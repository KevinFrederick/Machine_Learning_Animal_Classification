import numpy as np
import tensorflow as tf
import keras.utils as image
from tensorflow.keras.models import load_model

model = load_model('animal_classifier_(91).h5')
class_names = [
    'Butterfly',
    'Cat',
    'Cow',
    'Dog',
    'Elephant',
    'Hen',
    'Horse',
    'Monkey',
    'Panda',
    'Sheep',
    'Spider',
    'Squirrel'
]

# Load and preprocess image
def preprocess(imgList):
    processedImg = []

    for i in imgList:
        img_path = f'img/{i}'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x/255, axis=0)
        processedImg.append(x)

    return processedImg

# Predict image class
def predict(processedImg, imgList):
    preds = model.predict(np.vstack(processedImg))
    decode = [class_names[np.argmax(pred)] for pred in preds]
    result = []
    for i in range(len(processedImg)):
        predicted = {"Image" : f"img/{imgList[i]}", "Predicted" : decode[i]}
        result.append(predicted)
    return result