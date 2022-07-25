import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv

from helpers import dsc, iou, precision, recall
from boundary_iou import boundary_iou
import train

def calculate_metrics(predicted_ys, gt_ys):
  dscs = np.array([dsc(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  ious = np.array([iou(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  bious = np.array([boundary_iou(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  precisions = np.array([precision(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  recalls = np.array([recall(predicted_ys[i], gt_ys[i]) for i in range(len(gt_ys))])
  return dscs, ious, bious, precisions, recalls

def print_metrics(metrics):
  dscs, ious, bious, precisions, recalls = metrics
  print(f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | bIoU: {bious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')

def run_prediction(model, x, device, dataset):
  x = x.to(device)
  x = F.interpolate(x.unsqueeze(0), dataset.input_size)
  prediction = model(x)
  prediction = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
  return prediction

def get_predictions(model, dataset, device):
  all_ys = []
  all_predicted_ys = []

  for i in range(len(dataset)):
    _, y, _ = dataset.get_file_data(i)
    y = y.squeeze().detach().cpu().numpy()

    x, _ = dataset[i]
    y_pred = run_prediction(model, x, device, dataset)
    
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
