import argparse
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

def process_image(image):
    image_size = 224

    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image


def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    
    result = model.predict(image)
    result_sorted = np.sort(result[0])
    des_result_sorted = result_sorted[::-1]

    probs = np.array([])
    classes = np.zeros(top_k)
    
    for x in range(top_k):
        probs = np.append(probs, des_result_sorted[x] )
        classes[x] = int(np.where(result == des_result_sorted[x])[1])
    
    classes = np.int_(classes)
    return probs, classes




parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument(action="store", dest="model")
parser.add_argument(action="store", dest="image_path")

parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=3)
parser.add_argument('--category_names', action="store", dest="category_names", default="")

param = parser.parse_args()

model = tf.keras.models.load_model(param.model,custom_objects={'KerasLayer':hub.KerasLayer})

probs, classes = predict(param.image_path, model, param.top_k)

print(probs)


class_names = dict()

if(param.category_names == ""):
    print(classes)
else:
    with open(param.category_names, 'r') as f:
        class_names = json.load(f)
    class_names_new = dict()
    for key in class_names:
        class_names_new[str(int(key) -1)] = class_names[key]
    label_names = [class_names_new[str(id1)] for id1 in classes]
    print(label_names)

