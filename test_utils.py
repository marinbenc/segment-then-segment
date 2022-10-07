import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

from helpers import dsc, iou, precision, recall, _thresh
from boundary_iou import boundary_iou
import train

def small_objects_img(img, obj_size_percent):
  img_w, img_h = img.shape[-2:]
  img_area = img_w * img_h
  obj_size_th = img_area * obj_size_percent

  new_img = np.zeros_like(img)
  labeled = label(_thresh(img))
  regions = regionprops(labeled)
  for i in range(1, labeled.max()):
    if regions[i].convex_area < obj_size_th:
      new_img[labeled == regions[i].label] = 1

  # plt.imshow(img)
  # plt.show()

  # plt.imshow(new_img)
  # plt.show()

  return new_img

def _segmentation_metrics(predicted_ys, gt_ys):
  dscs = np.array([dsc(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  ious = np.array([iou(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  bious = np.array([boundary_iou(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  precisions = np.array([precision(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  recalls = np.array([recall(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  return dscs, ious, bious, precisions, recalls


def calculate_metrics(predicted_ys, gt_ys):

  metrics_all = _segmentation_metrics(predicted_ys, gt_ys)

  small_predicted_ys_all = [small_objects_img(img, 0.1) for img in predicted_ys]
  small_ys_all = [small_objects_img(img, 0.001) for img in gt_ys]

  # remove empty images
  small_predicted_ys = []
  small_ys = []
  for i in range(len(small_ys_all)):
    if np.any(small_ys_all[i]):
      small_ys.append(small_ys_all[i])
      small_predicted_ys.append(small_predicted_ys_all[i])

  metrics_small = _segmentation_metrics(small_predicted_ys, small_ys)

  return metrics_all, metrics_small

def print_metrics(metrics):
  dscs, ious, bious, precisions, recalls = metrics[0]
  print(f'DSC: {dscs.mean():.3f} \\pm {dscs.std():.3f} | IoU: {ious.mean():.3f}  \\pm {ious.std():.3f} | prec: {precisions.mean():.4f} \\pm {precisions.std():.3f} | rec: {recalls.mean():.4f}  \\pm {recalls.std():.3f}')

def run_prediction(model, x, device, dataset, original_size):
  x = x.to(device)
  x = F.interpolate(x.unsqueeze(0), dataset.input_size)
  prediction = model(x, output_size=original_size)
  prediction = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
  return prediction

def get_predictions(model, dataset, device):
  all_ys = []
  all_predicted_ys = []

  for i in range(len(dataset)):
    _, y, _ = dataset.get_file_data(i)
    y = y.squeeze().detach().cpu().numpy()

    x, _ = dataset[i]
    y_pred = run_prediction(model, x, device, dataset, original_size=y.shape[-2:])
    
    all_ys.append(y)
    all_predicted_ys.append(y_pred)

  return all_ys, all_predicted_ys

def get_model(weights, args, dataset_class, device):
  model = train.get_model(args, dataset_class, device)
  model.to(device)
  model.load_state_dict(torch.load(weights))
  model.eval()
  model.train(False)
  return model
