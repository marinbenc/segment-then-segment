import train
from types import SimpleNamespace
import os.path as p

def make_args(name, model, dataset, batch_size, lr, epochs, input_size, cropped):
  args = {
    'batch_size': batch_size,
    'epochs': epochs,
    'lr': lr,
    'model': model,
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
all_input_sizes = {
  'unet': [32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512],
  'resunetpp': [32, 64, 128, 256, 384],
  'deeplab': [32, 64, 128, 256, 384],
}


for model in train.model_choices:
  for base_params in params:
    print(base_params)
    params_copy = base_params.copy()
    input_sizes = [size for size in all_input_sizes[model] if size <= params_copy['max_size']]
    del params_copy['max_size']
    
    uncropped_params = [{
      'name': f'{params_copy["dataset"]}_uncropped_{s}_{model}',
      'model': model,
      'input_size': s,
      'cropped': False,
      **params_copy
    } for s in input_sizes]

    params_copy['epochs'] //= 2 # cropped models train faster
    cropped_params = [{
      'name': f'{params_copy["dataset"]}_cropped_{s}_{model}',
      'model': model,
      'input_size': s,
      'cropped': True,
      **params_copy
    } for s in input_sizes]

    for u_params in uncropped_params:
      args = make_args(**u_params)
      args = SimpleNamespace(**args)
      if p.exists(p.join('logs', args.experiment_name)):
        print(args.experiment_name, 'exists, skipping...')
        continue
      train.main(args)

    for c_params in cropped_params:
      args = make_args(**c_params)
      args = SimpleNamespace(**args)
      if p.exists(p.join('logs', args.experiment_name)):
        print(args.experiment_name, 'exists, skipping...')
        continue
      train.main(args)
