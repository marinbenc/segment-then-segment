import argparse
import sys
import os.path as p
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import train

from test_utils import get_predictions, run_prediction, get_model, calculate_metrics, print_metrics

def get_predictions_cropped(model, dataset, proposed_ys, device):
  all_predicted_ys = []
  all_xs = []

  for i in range(len(dataset)):
    x, y, _ = dataset.get_file_data(i)
    y = y[0].detach().cpu().numpy()
    
    y_pred = run_prediction_cropped(x, proposed_ys[i], y, model, dataset, device)    
    all_predicted_ys.append(y_pred)

    if x.shape[0] == 3:
      x = x.detach().cpu().numpy()
      x = np.moveaxis(x, 0, -1)
    else:
      x = x[0].detach().cpu().numpy()
    all_xs.append(x)

  return all_xs, all_predicted_ys

def run_prediction_cropped(x, y_proposal, y, model, dataset, device):
  y_proposal[y_proposal >= 0.5] = 1
  y_proposal[y_proposal < 0.5] = 0
  crops = dataset.get_crops_for_proposal(x, x, y_proposal)
  y_pred = np.zeros_like(y)
  
  for (x_crop, _, bbox) in crops:
    l, t, w, h = bbox
    if l + w > y.shape[-1] or t + h > y.shape[-2]:
      print(l, t, l + w, t + h, y.shape)
      continue

    y_pred_crop = run_prediction(model, x_crop, device, dataset, original_size=(w, h))
    y_pred_crop[y_pred_crop < 0.5] = 0
    y_pred_crop[y_pred_crop >= 0.5] = 1
    y_pred[t:t+h, l:l+w] = np.logical_or(y_pred[t:t+h, l:l+w], y_pred_crop)

    # plt.imshow(y[t:t+h, l:l+w])
    # plt.show()
    # plt.imshow(y_pred_crop)
    # plt.show()

  return y_pred

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  
  dataset_class = train.get_dataset_class(args)
  dataset = dataset_class('valid', cropped=False, input_size=args.input_size)
  model = get_model(args.weights, args, dataset_class, device)
  cropped_model = get_model(args.weights_cropped, args, dataset_class, device)

  time_start = timer()
  all_ys, all_predicted_ys = get_predictions(model, dataset, device)
  time_uncropped = timer()
  all_xs, all_predicted_ys_cropped = get_predictions_cropped(cropped_model, dataset, all_predicted_ys, device)
  time_cropped = timer()

  print('time uncropped: ', (time_uncropped - time_start) / len(dataset))
  print('time cropped: ', (time_cropped - time_start) / len(dataset))

  metrics_uncropped = calculate_metrics(all_predicted_ys, all_ys)
  metrics_cropped = calculate_metrics(all_predicted_ys_cropped, all_ys)
  # dscs = metrics[0]
  # plt.hist(dscs, bins=100)
  # plt.ylabel('DSC')
  # plt.xlabel('f')
  # plt.show()

  print(" cropped:")
  print_metrics(metrics_cropped)
  print(" uncropped:")
  print_metrics(metrics_uncropped)
  
  # sorting = np.argsort(dscs)[::-1]
  # for idx in sorting:
  #   print(all_ys[idx].shape)
  #   plt.imshow(all_ys[idx])
  #   plt.show()
  #   plt.imshow(all_predicted_ys_cropped[idx])
  #   plt.show()
  
  # return dscs.mean(), ious.mean(), precisions.mean(), recalls.mean()
  return all_xs, all_ys, all_predicted_ys, all_predicted_ys_cropped, metrics_uncropped, metrics_cropped
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Testing'
  )
  parser.add_argument(
    '--weights', type=str, help='path to weights'
  )
  parser.add_argument(
    '--weights-cropped', type=str, help='path to weights'
  )
  parser.add_argument(
    '--model', type=str, choices=train.model_choices, default='liver', help='dataset type'
  )
  parser.add_argument(
    '--dataset', type=str, choices=train.dataset_choices, default='liver', help='dataset type'
  )
  parser.add_argument(
      '--cropped', 
      action='store_true',
      help='use crops')
  parser.add_argument(
      '--input-size',
      type=int,
      default=256,
      help='size of input image, in pixels',
  )
  args = parser.parse_args()
  main(args)

