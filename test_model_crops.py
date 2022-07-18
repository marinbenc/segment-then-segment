import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
from skimage.filters import threshold_otsu

import train
from helpers import dsc, iou, precision, recall

def run_prediction(model, x, device):
  x = x.to(device)
  prediction = model(x.unsqueeze(0).detach())
  prediction = prediction.squeeze().detach().cpu()
  return prediction

def get_predictions(model, dataset, device):
  all_ys = []
  all_predicted_ys = []

  for i in range(len(dataset)):
    _, y, _ = dataset.get_file_data(i)
    y = y.squeeze().detach().cpu().numpy()

    x, _ = dataset[i]
    y_pred = run_prediction(model, x, device)
    
    all_ys.append(y)
    all_predicted_ys.append(y_pred)

  return all_ys, all_predicted_ys

def get_predictions_cropped(model, dataset, proposed_ys, device):
  all_predicted_ys = []

  for i in range(len(dataset)):
    x, y, _ = dataset.get_file_data(i)
    y = y[0].detach().cpu().numpy()
    
    y_pred = run_prediction_cropped(x, proposed_ys[i], y, model, dataset, device)    
    all_predicted_ys.append(y_pred)

  return all_predicted_ys


def run_prediction_cropped(x, y_proposal, y, model, dataset, device):
  y_proposal = cv.resize(y_proposal.numpy(), y.shape[-2:], cv.INTER_LINEAR)
  crops = dataset.get_crops_for_proposal(x, x, y_proposal)
  y_pred = np.zeros_like(y)

  for (x_crop, _, bbox) in crops:
    l, t, w, h = bbox

    y_pred_crop = run_prediction(model, x_crop, device).squeeze().numpy()
    y_pred_crop = cv.resize(y_pred_crop, (w, h), cv.INTER_LINEAR)

    y_pred_crop[y_pred_crop < 0.5] = 0
    y_pred_crop[y_pred_crop >= 0.5] = 1
    y_pred[t:t+h, l:l+w] = np.logical_or(y_pred[t:t+h, l:l+w], y_pred_crop)
  
  return y_pred

def get_model(weights, args, dataset_class, device):
  model = train.get_model(args, dataset_class, device)
  model.to(device)
  model.load_state_dict(torch.load(weights))
  model.eval()
  model.train(False)
  return model

def lerp(a, b, c):
  return c * a + (1 - c) * b

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  
  dataset_class = train.get_dataset_class(args)
  dataset = dataset_class('valid', cropped=False)
  model = get_model(args.weights, args, dataset_class, device)
  cropped_model = get_model(args.weights_cropped, args, dataset_class, device)

  all_ys, all_predicted_ys = get_predictions(model, dataset, device)
  all_predicted_ys_cropped = get_predictions_cropped(cropped_model, dataset, all_predicted_ys, device)

  dscs = np.array([dsc(all_predicted_ys_cropped[i], all_ys[i]) for i in range(len(all_ys))])
  ious = np.array([iou(all_predicted_ys_cropped[i], all_ys[i]) for i in range(len(all_ys))])
  precisions = np.array([precision(all_predicted_ys_cropped[i], all_ys[i]) for i in range(len(all_ys))])
  recalls = np.array([recall(all_predicted_ys_cropped[i], all_ys[i]) for i in range(len(all_ys))])

  # plt.hist(dscs, bins=100)
  # plt.ylabel('DSC')
  # plt.xlabel('f')
  # plt.show()

  print(f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')
  
  sorting = np.argsort(dscs)[::-1]
  for idx in sorting:
    plt.imshow(all_ys[idx])
    plt.show()
    plt.imshow(all_predicted_ys_cropped[idx])
    plt.show()
  
  return dscs.mean(), ious.mean(), precisions.mean(), recalls.mean()

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

  args = parser.parse_args()
  main(args)

