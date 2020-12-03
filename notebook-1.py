# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

import os
from tqdm import tqdm
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


# %% [markdown]
## Setting up paths
# %%
train_normal = os.path.join('../project_image_data/train/NORMAL')
train_pneumonia = os.path.join('../project_image_data/train/PNEUMONIA')


test_normal = os.path.join('../project_image_data/test/NORMAL')
test_pneumonia = os.path.join('../project_image_data/test/PNEUMONIA')

val_normal = os.path.join('../project_image_data/val/NORMAL')
val_pneumonia = os.path.join('../project_image_data/val/PNEUMONIA')

all_paths = [train_normal, train_pneumonia, test_normal, test_pneumonia, val_normal, val_pneumonia]
# %%
train_normal_files = [file for file in os.listdir(train_normal) if file.endswith('.jpeg')]
train_pneumonia_files = [file for file in os.listdir(train_pneumonia) if file.endswith('.jpeg')]

# test_normal_files = os.listdir(test_normal)
# test_pneumonia_files = os.listdir(test_pneumonia)

# val_normal_files = os.listdir(val_normal)
# val_pneumonia_files = os.listdir(val_pneumonia)
# %%
train_normal_files[:5]
# %%
for path in all_paths:
    print(f'{path} has  {len(os.listdir(path))}  files')
# %% [markdown]
## Showing Images
# %%
nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 10, nrows * 10)

pic_index += 8
next_normal_pix = [os.path.join(train_normal, fname) 
                for fname in train_normal_files[pic_index-8:pic_index]]
next_pneumonia_pix = [os.path.join(train_pneumonia, fname) 
                for fname in train_pneumonia_files[pic_index-8:pic_index]]

for i, img_path in enumerate(next_normal_pix+next_pneumonia_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
# %%
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# %%
model.summary()
# %%
 
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
# %%
# All images will be rescaled by 1./255
img_datagen = ImageDataGenerator(rescale=1./255)
 
# Flow training images in batches of 128 using train_datagen generator
train_generator = img_datagen.flow_from_directory(
        '../project_image_data/train/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

val_generator = img_datagen.flow_from_directory('../project_image_data/val/', 
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
# %%
val_generator
# %%
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=8, #epochs=15
      verbose=1,
      validation_data=val_generator)

# %%
res_df = pd.DataFrame(history.history)
res_df
# %%
fig = px.line(res_df, x=res_df.index, y=["accuracy","val_accuracy"])
fig.update_layout(title='Accuracy and Validation Accuracy over Epochs',
                  xaxis_title='Epoch',
                  yaxis_title='Percentage')
fig.show()
# %%
fig = px.line(res_df, x=res_df.index, y=['loss','val_loss'])
fig.update_layout(title='Loss and Validation Loss over Epochs',
                  xaxis_title='Epoch',
                  yaxis_title='idek what this unit is - change me')
fig.show()
# %%

# %%
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
accuracy=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
 
epochs=range(len(accuracy)) # Get number of epochs
 
#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, accuracy, 'r', "Training Accuracy", label='T-Accuracy')
plt.plot(epochs, val_acc, 'b', "Validation Accuracy", label='V-Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
 
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss", label='T-Loss')
plt.plot(epochs, val_loss, 'b', "Validation Loss", label='V-Loss')
plt.legend()
plt.figure()
# %%

# %%

# %%
