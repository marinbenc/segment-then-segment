import train
from types import SimpleNamespace

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

cells_input_sizes = [32, 64, 128, 256]
cells_uncropped_params = [{
  'name': f'cells_uncropped_{s}',
  'dataset': 'cells',
  'batch_size': 16,
  'lr': 0.0005,
  'epochs': 100,
  'input_size': s,
  'cropped': False
} for s in cells_input_sizes]

for params in cells_uncropped_params:
  args = make_args(**params)
  args = SimpleNamespace(**args)
  train.main(args)
