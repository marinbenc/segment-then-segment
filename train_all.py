import train
from types import SimpleNamespace
import os.path as p

def make_args(name, dataset, batch_size, lr, epochs, input_size, cropped):
  args = {
    'batch_size': batch_size,
    'epochs': epochs,
    'lr': lr,
    'model': 'unet',
    'loss': 'dsc',
    'dataset': dataset,
    'workers': 0,
    'input_size': input_size,
    'cropped': cropped,
    'experiment_name': name,
    'logs': './logs'
  }
  return args

cells_base_params = {
  'dataset': 'cells',
  'batch_size': 16,
  'lr': 0.0005,
  'epochs': 100,
  'max_size': 256
}

polyp_base_params = {
  'dataset': 'polyp',
  'batch_size': 8,
  'lr': 0.001,
  'epochs': 175,
  'max_size': 256
}

aorta_base_params = {
  'dataset': 'aa',
  'batch_size': 8,
  'lr': 0.001,
  'epochs': 100,
  'max_size': 512
}

params = [cells_base_params, polyp_base_params, aorta_base_params]
all_input_sizes = [32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512]

for base_params in params:
  input_sizes = [size for size in all_input_sizes if size <= base_params['max_size']]
  del base_params['max_size']

  print(base_params['dataset'], input_sizes)
  
  uncropped_params = [{
    'name': f'{base_params["dataset"]}_uncropped_{s}',
    'input_size': s,
    'cropped': False,
    **base_params
  } for s in input_sizes]
  cropped_params = [{
    'name': f'{base_params["dataset"]}_cropped_{s}',
    'input_size': s,
    'cropped': True,
    **base_params
  } for s in input_sizes]

  for params in uncropped_params:
    args = make_args(**params)
    args = SimpleNamespace(**args)
    if p.exists(p.join('logs', args.experiment_name)):
      print(args.experiment_name, 'exists, skipping...')
      continue
    train.main(args)

  for params in cropped_params:
    args = make_args(**params)
    args = SimpleNamespace(**args)
    if p.exists(p.join('logs', args.experiment_name)):
      print(args.experiment_name, 'exists, skipping...')
      continue
    train.main(args)
