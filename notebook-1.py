# %%
import os
import sys
import datetime
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objs as go
import plotly.express as px

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
# %load_ext tensorboard


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
    labels_arr = ['Normal' if label == 0 else 'Pneumonia' for label in labels_arr]
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, label, ax in zip( images_arr, labels_arr, axes):
        ax.imshow(img)
        ax.set_title(label, size=18)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['Normal','Pneumonia']

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def prepare_confusion_matrix(model):
    predictions = model.predict(x=test_generator, verbose=2)
    np.round(predictions)
    cm = confusion_matrix(y_true=test_generator.classes, y_pred=np.argmax(predictions, axis=-1))
    return cm
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

# %% [markdown]
## Showing Images
# %%
# All images will be rescaled by 1./255
img_datagen = ImageDataGenerator(rescale=1./255)
 
# Flow training images in batches of 128 using train_datagen generator
train_generator = img_datagen.flow_from_directory('../project_image_data/train/',  # Source dir for training images
                                                  target_size=(128, 128),  # All images will be resized to 150x150
                                                  batch_size=128,
                                                  color_mode='grayscale',
                                                  # Since we use binary_crossentropy loss, we need binary labels
                                                  class_mode='binary')

val_generator = img_datagen.flow_from_directory('../project_image_data/val/', # This is th source dir for validation images
                                                 target_size=(128, 128),  # All images will be resized to 150x150
                                                 batch_size=128,
                                                 color_mode='grayscale',
                                                 # Since we use binary_crossentropy loss, we need binary labels
                                                 class_mode='binary')

test_generator = img_datagen.flow_from_directory('../project_image_data/test/', # This is th source dir for validation images
                                                 target_size=(128, 128),  # All images will be resized to 150x150
                                                 batch_size=128,
                                                 color_mode='grayscale',
                                                 # Since we use binary_crossentropy loss, we need binary labels
                                                 class_mode='binary',
                                                 shuffle=False)       
# %%
X_test, y_test = next(test_generator)
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

# %%
imgs, labels = next(train_generator)
# %%

# %%
plotImages(imgs, labels)
print()
# %%
test_imgs, test_labels = next(test_generator)
plotImages(test_imgs, test_labels)
print(test_labels[:10])
# %%
# test_generator.classes

# %%
test_generator.class_indices

# %%
model1 = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
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
# %%
model1.summary()
# %%
%%time
model1.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
# %%
%%time
history1 = model1.fit(
      train_generator,
    #   steps_per_epoch=8,  
      epochs=5, #epochs=15
      verbose=1,
      validation_data=val_generator)

# %%
'''cm_plot_labels = ['Normal','Pneumonia']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')'''
# %%

#%%
plot_results(history1.history)
# %%
cm = prepare_confusion_matrix(model1)
plot_confusion_matrix(cm)

# %%
'''predictions = model1.predict(x=test_generator, verbose=2)
np.round(predictions)
cm = confusion_matrix(y_true=test_generator.classes, y_pred=np.argmax(predictions, axis=-1))'''
# %%
'''from sklearn import metrics
print(metrics.classification_report(y_test, predictions.round()))'''
# %%

# %%

model2 = tf.keras.models.Sequential([
  
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
      # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
        # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1

    tf.keras.layers.Dense(1, activation='sigmoid')
])
# %%
model2.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history2 = model2.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=20, #epochs=15
      verbose=1,
      validation_data=val_generator)

plot_results(history2.history)

# %%
model3 = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# %%
model3.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy'])

history3 = model3.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=1, #epochs=20
      verbose=1,
      validation_data=val_generator)

plot_results(history3.history)
# %%
tensorboard = TensorBoard(log_dir=f'./logs/model4_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
# %%


model4 = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])
# %%
%%time

model4.compile(loss='binary_crossentropy',
            optimizer="Adam",
            metrics=['accuracy'])

                                                    
history4 = model4.fit(
    train_generator,
    steps_per_epoch=8, #8  
    epochs=20, #epochs=20
    verbose=1,
    validation_data=val_generator,
    callbacks=[tensorboard]
    )

plot_results(history4.history)

# %% [markdown]
### Use LIME to show 'feature' selection in a sense
# %%

# %%
