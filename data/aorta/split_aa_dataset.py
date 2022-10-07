'''
Loads all of the scans from the data/ folder and saves each slice as a separate
numpy array file into the appropriate folder (e.g. train/<patient number>-<slice number>.npy) 
in the currrent directory.
'''

import os
import os.path as p
import sys

import matplotlib.pyplot as plt
import numpy as np
import nrrd
import cv2 as cv

sys.path.append('../../')
import helpers as h

def read_scan(file_path):
  ''' Read scan with axial view '''
  data, _ = nrrd.read(file_path)
  scan = np.rot90(data)
  scan = scan.astype(np.int16)
  return scan

scans_directory = 'data/avt/'

all_files = h.listdir(scans_directory)
all_files.sort()
label_files = [f for f in all_files if 'seg' in f]
np.random.seed(2022)
np.random.shuffle(label_files)
print(label_files)

folders = ('train', 'valid', 'test')

split0 = int(len(label_files) * 0.7)
split1 = split0 + int(len(label_files) * 0.2)

split_label_files = (label_files[:split0], label_files[split0:split1], label_files[split1:])
for split in split_label_files:
  print(len(split))

for (folder, labels) in zip(folders, split_label_files):
  h.mkdir(folder)
  h.mkdir(p.join(folder, 'input'))
  h.mkdir(p.join(folder, 'label'))

  for mask_file in labels:
    volume_file = mask_file.replace('seg.', '')
    volume_scan = read_scan(p.join(scans_directory, volume_file))
    mask_scan = read_scan(p.join(scans_directory, mask_file))

    for i in range(mask_scan.shape[-1]):
      mask_slice = mask_scan[..., i]
      if mask_slice.sum() <= 0:
        # skip empty slices
        continue

      volume_slice = volume_scan[..., i]
      original_mask_name = mask_file.split('.')[0]

      volume_name = f'{original_mask_name}-{i}.npy'
      mask_name = f'{original_mask_name}-{i}.npy'

      volume_save_path = p.join(folder, 'input', volume_name)
      mask_save_path = p.join(folder, 'label', mask_name)
      print(volume_name, volume_slice.dtype)

      np.save(volume_save_path, volume_slice)
      np.save(mask_save_path, mask_slice)