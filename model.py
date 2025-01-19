import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Path to Model File
model = load_model('animal_classifier_(91).h5')
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
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

def preprocess(img):
    img = Image.open(img)
    x = img.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    x = np.array(x)
    x = np.expand_dims(x / 255, axis=0)
    return x

def predict(processedImg, img):
    pred = model.predict(processedImg)
    predicted_label = np.argmax(pred)
    label_name = class_names[predicted_label]
    result = {'image': img, 'prediction': label_name}
    return result