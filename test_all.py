from types import SimpleNamespace
import os.path as p
import json
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import test
import test_model_crops
import helpers as h

def load_training_args(folder):
  args_file = p.join(folder, 'args.json')
  args = json.load(args_file)
  return args

def get_best_model(folder):
  files = h.listdir(folder)
  files = [f for f in files if 'best_model_' in f]
  files = natsorted(files)
  return files[-1]

def make_args(weights_folder, dataset, input_size, cropped):
  weights = p.join(weights_folder, get_best_model(weights_folder))
  args = {
    'weights': weights,
    'model': 'unet',
    'dataset': dataset,
    'cropped': cropped,
    'input_size': input_size
  }
  return args

def make_args_cropped(uncropped_weights_folder, cropped_weights_folder, dataset, input_size, cropped):
  uncropped_weights = p.join(uncropped_weights_folder, get_best_model(uncropped_weights_folder))
  cropped_weights = p.join(cropped_weights_folder, get_best_model(cropped_weights_folder))
  args = {
    'weights': uncropped_weights,
    'weights_cropped': cropped_weights,
    'model': 'unet',
    'dataset': dataset,
    'cropped': cropped,
    'input_size': input_size
  }
  return args

datasets = ['cells', 'polyp', 'aa']
input_sizes = [32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512]

for dataset in datasets:
  print(dataset)

  folders_uncropped = []
  folders_cropped = []

  for input_size in input_sizes:
    uncropped_folder = p.join('logs', f'{dataset}_uncropped_{input_size}')
    folders_uncropped.append(uncropped_folder)
    cropped_folder = p.join('logs', f'{dataset}_cropped_{input_size}')
    folders_cropped.append(cropped_folder)

  for (folder_uncropped, folder, input_size) in zip(folders_uncropped, folders_cropped, input_sizes):
    if p.exists(folder):
      args = make_args_cropped(folder_uncropped, folder, dataset, input_size, True)
      print(folder)
      args = SimpleNamespace(**args)
      
      all_xs, all_ys, all_predicted_ys, all_predicted_ys_cropped, metrics_uncropped, metrics_cropped = test_model_crops.main(args)
      np.save(p.join(folder, 'output_metrics_cropped.npy'), np.array(metrics_cropped[0]))
      np.save(p.join(folder, 'output_metrics_uncropped.npy'), np.array(metrics_uncropped[0]))
      if input_size == 64 or input_size == 128 or input_size == 256 or input_size == 512:
        print('Saving images...')
        predictions_folder = p.join(folder, 'predictions', str(input_size))
        h.mkdir(predictions_folder)
        for i in range(len(all_xs)):
          if dataset == 'aa':
            all_xs[i] *= 255
          cv.imwrite(p.join(predictions_folder, f'{i}_input.png'), all_xs[i])
          plt.imsave(p.join(predictions_folder, f'{i}_gt.png'), all_ys[i])
          plt.imsave(p.join(predictions_folder, f'{i}_cropped.png'), all_predicted_ys_cropped[i])
          plt.imsave(p.join(predictions_folder, f'{i}_uncropped.png'), all_predicted_ys[i])
