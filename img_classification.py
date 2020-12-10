import tensorflow as tf
from tensorflow import keras
# import keras
from PIL import Image, ImageOps
import numpy as np

weights_file = "./Adam_32_32_32_32__best"
# weights_file = "./checkpoints/testloadd.data-00000-of-00001"
def teachable_machine_classification(img, weights_file):
    # Load the model
    model = tf.keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 64, 64, ), dtype=np.float32)

    print(data)
    image = ImageOps.grayscale(img)
    #image sizing
    size = (64,64)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    print(normalized_image_array)
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability