# %%
import os
import sys
import datetime
import itertools
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.random import set_seed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import metrics

import visualkeras

# %load_ext tensorboard
set_seed(42)
np.random.seed(42)

# %%
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%
def plot_results(results):
  """Function to convert a models results into a dataframe and plot them to show the both the accuracy and validation accuracy, as well as the loss and validation loss over epochs.

  Args:
      results_dataframe (dataframe): 
  """

  results_dataframe = pd.DataFrame(results)

  fig = px.line(results_dataframe, x=results_dataframe.index, y=["accuracy","val_accuracy"])
  fig.update_layout(title='Accuracy and Validation Accuracy over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Percentage',
                )
  fig.update_traces(mode='lines+markers')
  fig.show()

  fig = px.line(results_dataframe, x=results_dataframe.index, y=['loss','val_loss'])
  fig.update_layout(title='Loss and Validation Loss over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='idek what this unit is - change me'
                )
  fig.update_traces(mode='lines+markers')
  fig.show()

def plotImages(images_arr, labels_arr):
    labels_arr = ['Normal: 0' if label == 0 else 'Pneumonia: 1' for label in labels_arr]
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, label, ax in zip( images_arr, labels_arr, axes):
        ax.imshow(img)
        ax.set_title(label, size=18)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_results(model): 
    predictions = model.predict(X_test).round()
    cm = metrics.confusion_matrix(y_test, predictions,
                                normalize='true')

    ax = sns.heatmap(cm, cmap='Greens',annot=True,square=True)
    ax.set(xlabel='Predicted Class',ylabel='True Class')
    print(metrics.classification_report(y_test, predictions))

# def prepare_confusion_matrix(model):

#     predictions = model.predict(X_test).round()
#     cm = confusion_matrix(y_test, predictions)
#     return cm, predictions
    # test_generator.class_indices
  
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

'''train_normal_files = [file for file in os.listdir(train_normal) if file.endswith('.jpeg')]
train_pneumonia_files = [file for file in os.listdir(train_pneumonia) if file.endswith('.jpeg')]
'''

# %%
'''train_normal_files[:5]
'''
# %%

for path in all_paths:
    print(f'{path} has  {len(os.listdir(path))}  files')

# %%

# %% [markdown]
## Showing Images
# %%
# All images will be rescaled by 1./255
img_datagen = ImageDataGenerator(rescale=1./255)
 
# Flow training images in batches of 128 using train_datagen generator
train_generator = img_datagen.flow_from_directory('../project_image_data/train/',  # Source dir for training images
                                                  target_size=(64, 64),  # All images will be resized to 150x150
                                                  batch_size=2606, #128
                                                  color_mode='grayscale',
                                                  # Since we use binary_crossentropy loss, we need binary labels
                                                  class_mode='binary')

val_generator = img_datagen.flow_from_directory('../project_image_data/val/', # This is th source dir for validation images
                                                 target_size=(64, 64),  # All images will be resized to 150x150
                                                 batch_size=92, #128
                                                 color_mode='grayscale',
                                                 # Since we use binary_crossentropy loss, we need binary labels
                                                 class_mode='binary')

test_generator = img_datagen.flow_from_directory('../project_image_data/test/', # This is th source dir for validation images
                                                 target_size=(64, 64),  # All images will be resized to 150x150
                                                 batch_size=624, #128
                                                 color_mode='grayscale',
                                                 # Since we use binary_crossentropy loss, we need binary labels
                                                 class_mode='binary',
                                                 shuffle=False)       
# %%
X_train,y_train = next(train_generator)
X_test,y_test = next(test_generator)
X_val,y_val = next(val_generator)
# %%
'''nrows = 4
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

plt.show()'''


# %%
# X_train,y_train = next(train_generator)
# X_test,y_test = next(test_generator)
# X_val,y_val = next(val_generator)
# %%
plotImages(X_train, y_train)
print(y_train[:10])
print(X_train[0].shape)
# %%
# imgs, labels = next(train_generator)
# plotImages(imgs, labels)
# print()
# %%
# test_imgs, test_labels = next(X_test,y_test)
# plotImages(test_imgs, test_labels)
# print(test_labels[:10])
# %%
# test_generator.classes

# %%
test_generator.class_indices

# %%
# callbacks = [checkpoint, tensorboard, earlystop]
# %%

RMSprop_32_64 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
RMSprop_32_64.summary()
# %%
# %%
%%time
RMSprop_32_64.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy',tf.keras.metrics.Recall()])
# %%

# os.makedirs(filepath,exist_ok=True)
tensorboard = TensorBoard(log_dir=f'./logs/RMSprop_32_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
earlystop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=True)
# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/RMSprop_32_64', verbose=1, save_best_only=True, mode='auto')
history1 = RMSprop_32_64.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history1.history)
evaluate_results(RMSprop_32_64)

# %%
'''predictions = RMSprop_32_64.predict(x=test_generator, verbose=2)
np.round(predictions)
cm = confusion_matrix(y_true=test_generator.classes, y_pred=np.argmax(predictions, axis=-1))'''
# %%
'''from sklearn import metrics
print(metrics.classification_report(y_test, predictions.round()))'''
# %%
Adam_32_64 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adam_32_64.summary()
Adam_32_64.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_64_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history2 = Adam_32_64.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history2.history)
evaluate_results(Adam_32_64)

# %%
Adam_32_64_64 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adam_32_64_64.summary()
Adam_32_64_64.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])
# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_64_64_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_64_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history3 = Adam_32_64_64.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[tensorboard, earlystop, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history3.history)
evaluate_results(Adam_32_64_64)
# %%
Adam_32_64_128 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adam_32_64_128.summary()
Adam_32_64_128.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])
# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_64_128_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_64_128_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history4 = Adam_32_64_128.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history4.history)
evaluate_results(Adam_32_64_128)
# %%
Adam_32_64_128_256 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adam_32_64_128_256.summary()
Adam_32_64_128_256.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_64_128_256', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_64_128_256{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history5 = Adam_32_64_128_256.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history5.history)
evaluate_results(Adam_32_64_128_256)
# %%
Adam_32_64_64_64 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adam_32_64_64_64.summary()
Adam_32_64_64_64.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_64_64_64_', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_64_64_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history6 = Adam_32_64_64_64.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history6.history)
evaluate_results(Adam_32_64_64_64)
# %%
earlystop = tf.keras.callbacks.EarlyStopping(patience=5, verbose=True)
Adam_32_64_64_64_P5 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adam_32_64_64_64_P5.summary()
Adam_32_64_64_64_P5.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_64_64_64_P5_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_64_64_64_P5_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history7 = Adam_32_64_64_64_P5.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[tensorboard, earlystop, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history7.history)
evaluate_results(Adam_32_64_64_64_P5)
# %%
earlystop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=True)
Adam_32_32_32_32 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adam_32_32_32_32.summary()
Adam_32_32_32_32.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_32_32_32_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_32_32_32_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history8 = Adam_32_32_32_32.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[tensorboard, earlystop, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history8.history)
evaluate_results(Adam_32_32_32_32)
# %% [markdown]
### Use LIME to show 'feature' selection in a sense
# %%
Adam_32_32_32_32_2D = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(64, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# %%
Adam_32_32_32_32_2D.summary()
# %%
Adam_32_32_32_32_2D.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy',tf.keras.metrics.Recall()])

# %%
%%time
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/Adam_32_32_32_32_2D_', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_32_32_32_2D_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
history9 = Adam_32_32_32_32_2D.fit(
      X_train, 
      y_train,
      batch_size=128,
    #   steps_per_epoch=8,  
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))

# %%
plot_results(history9.history)
evaluate_results(Adam_32_32_32_32_2D)
# %%
# visualkeras.layered_view(Adam_32_32_32_32, to_file='network_visual.png').show()
# %%
testload = tf.keras.models.load_model('Adam_32_32_32_32__best')

# %%
earlystop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=True)
checkpoint = ModelCheckpoint(filepath=r'./checkpoints/testloadd.hdf5', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=f'./logs/Adam_32_64_64_64_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
test_HISTORY = testload.fit(
      X_train, 
      y_train,
    #   steps_per_epoch=8,  
      batch_size=128,
      epochs=100, #epochs=15
      verbose=1,
      callbacks=[earlystop, tensorboard, checkpoint],
      validation_data=(X_val, y_val))
# %%
visualkeras.layered_view(testload).show()

# %%
evaluate_results(testload)
# %%
preds = testload.predict(X_test)
fpr, tpr, thresh = roc_curve(y_test, preds)

# Calculate the ROC (Reciever Operating Characteristic) AUC (Area Under the Curve)
rocauc = auc(fpr, tpr)
print('Train ROC AUC Score: ', rocauc)
# %%
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
image = Image.open(r'D:\Python_Projects\flatiron\class-materials\phase04\project_image_data\test\PNEUMONIA\person1_virus_7.jpeg')
weights_file = "./Adam_32_32_32_32__best"

label = teachable_machine_classification(image, weights_file)
if label == 0:
    st.write("The image has pneumonia")
else:
    st.write("The image is healthy")
# %%
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
#     marker=dict(color='fa7f72'),

    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain', tickvals=[0,0.25,0.5,0.75,1])

fig.show()

# %%
successive_outputs = [layer.output for layer in testload.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = testload.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
# horse_img_files = [os.path.join(train_normal, f) for f in train_normal]
# human_img_files = [os.path.join(train_pneumonia, f) for f in train_normal]
# img_path = random.choice(horse_img_files + human_img_files)
img = load_img(train_pneumonia+'/person449_bacteria_1940.jpeg', target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
print(x)
# Rescale by 1/255
x /= 255
 
# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)
 
# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in testload.layers]
 
# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      if x.std()>0:
        x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
# %%
