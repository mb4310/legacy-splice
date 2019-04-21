import pickle
from pathlib import Path

"""
This file sets up the arguments for the experiment to run so we all you need to pass argparse is the name of the experiment.
"""

config = {
    'name':'CIFAR_01',
    'dataset':'CIFAR10',
    'initial_splice':[[64,1,0]],
    'splice':[[128,2,0], [128,2,0]],
    'early_stop':True,
    'batch_size':128,
    'image_size':112,
    'stop_percent':0.002,
    'seed':229,
    'save':False,
    'max_lr':7e-4,
    'num_epochs':10,
    'n_trials' : 3
}

path = Path('experiments/{}'.format(config['name']))
path.mkdir(exist_ok=True)
file_path = path/'config.pickle'

with open(file_path, 'wb') as handle:
    pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)


