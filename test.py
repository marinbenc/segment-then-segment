import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import train

from test_utils import get_predictions, run_prediction, get_model, calculate_metrics, print_metrics, small_objects_img

def main(args):
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  
  dataset_class = train.get_dataset_class(args)
  dataset = dataset_class('valid', cropped=args.cropped, input_size=args.input_size)
  model = get_model(args.weights, args, dataset_class, device)

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
        y_pred_crop = run_prediction(model, x, device, dataset)
        y_pred_crop = cv.resize(y_pred_crop, (w, h), cv.INTER_LINEAR)
        y_pred_crop[y_pred_crop < 0.5] = 0
        y_pred_crop[y_pred_crop >= 0.5] = 1
        y_pred[t:t+h, l:l+w] = np.logical_or(y_pred[t:t+h, l:l+w], y_pred_crop)

        # plt.imshow(y[t:t+h, l:l+w])
        # plt.show()
        # plt.imshow(y_pred_crop)
        # plt.show()
    else:
      x, _ = dataset[i]
      y_pred = run_prediction(model, x, device, dataset)
      y_pred = cv.resize(y_pred, y.shape[-2:][::-1], cv.INTER_LINEAR)
    
    all_ys.append(y)
    all_predicted_ys.append(y_pred)

  metrics = calculate_metrics(all_predicted_ys, all_ys)
  print_metrics(metrics)
  dscs = metrics[0]

  # plt.hist(dscs, bins=100)
  # plt.ylabel('DSC')
  # plt.xlabel('f')
  # plt.show()
  
  sorting = np.argsort(dscs)
  for idx in sorting:
    #print(bboxes[idx])
    plt.imshow(all_ys[idx])
    plt.show()
    # l, t, w, h = bboxes[idx][0]
    # viz_img = cv.rectangle(all_xs[idx], (l, t), (l + w, t + h), round(all_xs[idx].max()), 5)
    # plt.imshow(viz_img)
    # plt.show()
    plt.imshow(all_predicted_ys[idx])
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
  parser.add_argument(
      '--input-size',
      type=int,
      default=256,
      help='size of input image, in pixels',
  )
  args = parser.parse_args()
  main(args)

