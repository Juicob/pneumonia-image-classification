import tensorflow as tf
from tensorflow import keras
# import keras
from PIL import Image, ImageOps
import numpy as np

weights_file = "./Adam_32_32_32_32__best"
# weights_file = "./checkpoints/keras_model.h5"
# weights_file = "./model.savedmodel"
def teachable_machine_classification(img, weights_file):
    # Load the model
    model = tf.keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 64, 64, 1), dtype=np.float32)

    image = ImageOps.grayscale(img)
    #image sizing
    size = (64,64)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 32.0) - 1
    normalized_image_array = np.expand_dims(normalized_image_array, axis=2)

    data[0] = normalized_image_array


    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability