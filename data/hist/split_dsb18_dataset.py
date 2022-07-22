import shutil
import sys
import os.path as p

import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append('../..')
import helpers as h

original_folder = 'dsb18/stage1_train'
img_names = h.listdir(original_folder)

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
    shutil.copy(p.join(original_folder, img, 'images', img + '.png'), p.join(dst_folder, 'input'))

    # add all of the masks together
    masks = h.listdir(p.join(original_folder, img, 'masks'))
    mask_all = cv.imread(p.join(original_folder, img, 'masks', masks[0]), cv.IMREAD_GRAYSCALE)
    for mask_file in masks[1:]:
      mask = cv.imread(p.join(original_folder, img, 'masks', mask_file), cv.IMREAD_GRAYSCALE)
      mask_all = np.maximum(mask_all, mask)
    cv.imwrite(p.join(dst_folder, 'label', img + '.png'), mask_all)
