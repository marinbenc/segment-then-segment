import shutil
import sys
import os.path as p

import numpy as np

sys.path.append('../..')
import helpers as h

original_train = 'original_split/train'
original_test = 'original_split/test'

train_folder, valid_folder, test_folder = ['train', 'valid', 'test']
folders = [train_folder, valid_folder, test_folder]

for f in folders:
  h.mkdir(p.join(f, 'input'))
  h.mkdir(p.join(f, 'label'))

all_train_imgs = h.listdir(p.join(original_train, 'Images'))

np.random.seed(42)
np.random.shuffle(all_train_imgs)

split_idx = int(len(all_train_imgs) * 0.9)
train_imgs, valid_imgs = all_train_imgs[:split_idx], all_train_imgs[split_idx:]
test_imgs = h.listdir(p.join(original_test, 'Images'))

splits = [
  (train_imgs, original_train, train_folder), 
  (valid_imgs, original_train, valid_folder), 
  (test_imgs, original_test, test_folder)]

for imgs, src_folder, dst_folder in splits:
  for img in imgs:
    shutil.copy(p.join(src_folder, 'Images', img), p.join(dst_folder, 'input'))
    shutil.copy(p.join(src_folder, 'Labels', img).replace('.png', '.mat'), p.join(dst_folder, 'label'))
