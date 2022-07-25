import shutil
import sys
import os.path as p

import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append('../..')
import helpers as h

original_folder = 'Kvasir-SEG'
original_inputs = p.join(original_folder, 'images')
original_labels = p.join(original_folder, 'masks')
img_names = h.listdir(original_labels)

# (train, valid, test)
split = [int(percent * len(img_names)) for percent in [0.8, 0.1, 0.1]]
train_imgs, valid_imgs = train_test_split(img_names, test_size=split[1], random_state=42)
train_imgs, test_imgs = train_test_split(train_imgs, test_size=split[2], random_state=42)

train_folder, valid_folder, test_folder = ['train', 'valid', 'test']
folders = [train_folder, valid_folder, test_folder]

for f in folders:
  h.mkdir(p.join(f, 'input'))
  h.mkdir(p.join(f, 'label'))

splits = [
  (train_imgs, train_folder), 
  (valid_imgs, valid_folder), 
  (test_imgs, test_folder)]

for imgs, dst_folder in splits:
  for img in imgs:
    shutil.copy(p.join(original_inputs, img), p.join(dst_folder, 'input'))
    shutil.copy(p.join(original_labels, img), p.join(dst_folder, 'label'))