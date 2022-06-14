import argparse
import json
import os
import sys
import datetime

import segmentation_models_pytorch as smp

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, DiceCoefficient, MeanSquaredError
from ignite.contrib.metrics.regression import MedianAbsolutePercentageError

import helpers as h
from loss import DiceLoss
from dice_metric import DiceMetric

sys.path.append('saliency-sampler')
from saliency_sampler import Saliency_Sampler
from saliency_network import saliency_network_resnet18

from spatial_transformer import SpatialTransformer

sys.path.append('data')
from aa_dataset import AortaDataset

dataset_choices = ['aa']
model_choices = ['unet']

def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    dataset_class = get_dataset_class(args)
    loader_train, loader_valid = data_loaders(args, dataset_class)

    model = get_model(args, dataset_class, device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = DiceLoss()

    metrics = {
      'dsc': DiceMetric(device=device),
      'loss': Loss(criterion)
    }

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger('Train Evaluator')
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger('Val Evaluator')

    best_dsc = 0

    # @trainer.on(Events.GET_BATCH_COMPLETED(once=1))
    # def plot_batch(engine):
    #     x, y = engine.state.batch
    #     images = [x[0], y[0]]
    #     for image in images:
    #         if image.shape[0] > 1:
    #             image = image.numpy()
    #             image = image.transpose(1, 2, 0)
    #             image += 0.5
    #         plt.imshow(image.squeeze())
    #         plt.show()

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        nonlocal best_dsc
        train_evaluator.run(loader_train)
        validation_evaluator.run(loader_valid)
        curr_dsc = validation_evaluator.state.metrics['dsc']
        if curr_dsc > best_dsc:
            best_dsc = curr_dsc

    log_dir = f'logs/{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    tb_logger = TensorboardLogger(log_dir=log_dir)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='training',
        output_transform=lambda loss: {'batchloss': loss},
        metric_names='all',
    )

    for tag, evaluator in [('training', train_evaluator), ('validation', validation_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names='all',
            global_step_transform=global_step_from_engine(trainer),
        )

    def score_function(engine):
        return engine.state.metrics['dsc']

    model_checkpoint = ModelCheckpoint(
        log_dir,
        n_saved=2,
        filename_prefix='best',
        score_function=score_function,
        score_name='dsc',
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {'model': model})

    trainer.run(loader_train, max_epochs=args.epochs)
    tb_logger.close()

def get_dataset_class(args):
    mapping = {
      'aa':    AortaDataset,
    }
    return mapping[args.dataset]

def get_model(args, dataset_class, device):
    if args.model == 'unet':
        task_network = smp.Unet(in_channels=2, classes=1)
        model = SpatialTransformer(task_network)
    return model

def data_loaders(args, dataset_class):
    dataset_train, dataset_valid = datasets(args, dataset_class)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid

def datasets(args, dataset_class):
    train = dataset_class(
      folder='train',
    )
    valid = dataset_class(
      folder='valid',
    )
    return train, valid

def makedirs(args):
    os.makedirs(args.logs, exist_ok=True)

def snapshotargs(args):
    args_file = os.path.join(args.logs, 'args.json')
    with open(args_file, 'w') as fp:
        json.dump(vars(args), fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--logs', type=str, default='./logs', help='folder to save logs'
    )
    parser.add_argument(
        '--model', type=str, choices=model_choices, default='unet', help='which model architecture to use'
    )
    parser.add_argument(
        '--dataset', type=str, choices=dataset_choices, default='liver', help='which dataset to use'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of workers for data loading (default: 4)',
    )
    parser.add_argument(
        '--polar', 
        action='store_true',
        help='use polar coordinates')
    parser.add_argument(
        '--percent',
        type=float,
        default=None,
        help='percent of the training dataset to use',
    )
    args = parser.parse_args()
    main(args)