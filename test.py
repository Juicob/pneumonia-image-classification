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
    import tensorflow as tf

    model = tf.keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 64, 64, 1), dtype=np.float32)

    # print(data)
    image = ImageOps.grayscale(img)
    #image sizing
    size = (64,64)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)


    #turn the image into a numpy array
    image_array = np.asarray(image)
    # image_array = np.append(image_array.shape, [1])
    # image_array = image_array.reshape((1,) + image_array.shape)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 32.0) - 1
    normalized_image_array = np.expand_dims(normalized_image_array, axis=2)
    # print(normalized_image_array)
    # Load the image into the array
    # data = data.reshape((64, 64, 1))
    data[0] = normalized_image_array
    # data[0] = image_array
    # data[0] = np.array([64,64,1])

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

image = Image.open(r'D:\Python_Projects\flatiron\class-materials\phase04\project_image_data\test\PNEUMONIA\person1_virus_7.jpeg')
    # st.image(image, caption='Uploaded MRI.', use_column_width=True)
    # st.write("")
    # st.write("Classifying...")
label = teachable_machine_classification(image, weights_file)

print(label)