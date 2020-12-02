# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import os
from tqdm import tqdm
import tensorflow as tf

from tensorflow import keras

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
# train_normal_files = os.listdir(train_normal)
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

# %%

# %%

# %%
