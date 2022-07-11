import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
from skimage.filters import threshold_otsu

import train
from train import get_model
from helpers import dsc, iou, precision, recall

def run_prediction(model, x, device):
  x = x.to(device)
  prediction = model(x.unsqueeze(0).detach())
  prediction = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
  return prediction

def get_predictions(model, dataset, device):
  all_xs = []
  all_ys = []
  all_predicted_y1 = []
  #all_predicted_y2 = []
  #all_bboxes = []

  with torch.no_grad():
    for (x, y) in dataset:
      x = x.to(device)
      prediction = model(x.unsqueeze(0).detach())
      #all_bboxes.append(np.array(model.last_bounding_rects))

      #predicted_y1, predicted_y2 = prediction
      predicted_y1 = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
      #predicted_y2 = predicted_y2.squeeze(0).squeeze(0).detach().cpu().numpy()

      all_predicted_y1.append(predicted_y1)
      #all_predicted_y2.append(predicted_y2)

      x = x.squeeze(0).detach().cpu().numpy()
      all_xs.append(x)

      y = y.squeeze(0).detach().cpu().numpy()
      all_ys.append(y)

      # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
      # ax1.imshow(y)
      # ax1.set_title('GT')
      # ax2.imshow(predicted_y.squeeze())
      # ax2.set_title('Predicted')
      # plt.show()
  
  return all_xs, all_ys, all_predicted_y1#, all_predicted_y2, all_bboxes

def lerp(a, b, c):
  return c * a + (1 - c) * b

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  
  dataset_class = train.get_dataset_class(args)
  dataset = dataset_class('valid', cropped=args.cropped)

  model = get_model(args, dataset_class, device)
  model.to(device)
  model.load_state_dict(torch.load(args.weights))
  model.eval()
  model.train(False)

  all_ys = []
  all_predicted_ys = []

  for i in range(len(dataset)):
    _, y, _ = dataset.get_file_data(i)
    y = y.squeeze().detach().cpu().numpy()

    if args.cropped:
      y_pred = np.zeros_like(y)

      crops = dataset.get_crops(i)
      for (x, _, bbox) in crops:
        l, t, w, h = bbox
        y_pred_crop = run_prediction(model, x, device)
        y_pred_crop = cv.resize(y_pred_crop, (w, h), cv.INTER_LINEAR)

        y_pred_crop[y_pred_crop < 0.5] = 0
        y_pred_crop[y_pred_crop >= 0.5] = 1
        y_pred[t:t+h, l:l+w] = np.logical_or(y_pred[t:t+h, l:l+w], y_pred_crop)
    else:
      x, _ = dataset[i]
      y_pred = run_prediction(model, x, device)
      y_pred = cv.resize(y_pred, y.shape[-2:], cv.INTER_LINEAR)
    
    all_ys.append(y)
    all_predicted_ys.append(y_pred)

  dscs = np.array([dsc(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  ious = np.array([iou(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  precisions = np.array([precision(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
  recalls = np.array([recall(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])

  plt.hist(dscs, bins=100)
  plt.ylabel('DSC')
  plt.xlabel('f')
  plt.show()

  print(f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')
  
  sorting = np.argsort(dscs)
  for idx in sorting:
    print(bboxes[idx])
    plt.imshow(all_ys[idx])
    plt.show()
    l, t, w, h = bboxes[idx][0]
    viz_img = cv.rectangle(all_xs[idx], (l, t), (l + w, t + h), round(all_xs[idx].max()), 5)
    plt.imshow(viz_img)
    plt.show()
    thresh = all_predicted_y1[idx]
    plt.imshow(thresh > 0.05)
    plt.show()
    plt.imshow(all_predicted_y2[idx])
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

