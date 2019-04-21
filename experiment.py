import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import fastai
import fastai.vision
import time, copy
import core
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(description='Experiment Parameters')
parser.add_argument("-n", "--name", help="Enter the name for your experiment", default="TEST")
args = parser.parse_args()

path = Path('experiments')
exp_path = path/f'{args.name}'
config_path = exp_path/'config.pickle'

with open(config_path, 'rb') as handle:
    args = pickle.load(handle)

"Fetch the arguments based on the name of the experiment..."

ds = args['dataset']
image_size = args['image_size']
batchsize = args['batch_size']
initial_splice = args['initial_splice']
splice = args['splice']
early_stop = args['early_stop']
stop_pct = args['stop_percent']
seed = args['seed']
save = args['save']
max_lr = args['max_lr']
num_epochs = args['num_epochs']
n_trials = args['n_trials']

torch.manual_seed(seed)
np.random.seed(seed)

"""Loads in the data..."""

if ds == 'CIFAR10':
    url = fastai.datasets.URLs.CIFAR
    path = fastai.datasets.untar_data(url)
    data = (fastai.vision.ImageItemList.from_folder(path)
            .split_by_folder(train='train', valid='test')
            .label_from_folder()
            .transform(fastai.vision.get_transforms(), size=image_size)
            .databunch(bs=batchsize))
    
elif ds == 'MNIST':
    url = fastai.datasets.URLs.MNIST
    path = fastai.datasets.untar_data(url)
    data = fastai.vision.ImageDataBunch.from_folder(path, train='training', valid='testing', bs=batchsize)
    
elif ds == 'PETS':
    func = lambda x: str(x)[46:].rstrip('.jpg1234567890').rstrip('_')
    url = fastai.datasets.URLs.PETS
    path = fastai.datasets.untar_data(url)
    data = (fastai.vision.ImageItemList.from_folder(path/'images')
            .random_split_by_pct()
            .label_from_func(func)
            .transform(fastai.vision.get_transforms(), size=image_size)
            .databunch(bs=batchsize))

elif ds == 'FOOD':
    path = Path('/home/max/Desktop/datasets/food-101/images')
    data = (fastai.vision.ImageItemList.from_folder(path)
        .random_split_by_pct()
        .label_from_folder()
        .transform(fastai.vision.get_transforms(), size=image_size)
        .databunch(bs=batchsize))

else:
    raise('You must select a valid dataset for this experiment.')
    
crit = nn.CrossEntropyLoss()
dls = {'train':data.train_dl, 'val':data.valid_dl}
tr_histories = []
val_histories = []

for trial in range(n_trials):
    print('Starting trial {} at {}'.format(trial, time.ctime()))

    base_model = core.cnn(nc=data.c).cuda()
    base_model.splice(initial_splice)
    base_model.init_params()
    base_model.init_opts(max_lr = max_lr, pct_start = 0.10)

    comp_model = copy.deepcopy(base_model)
    comp_model.splice(splice)
    comp_model.init_params()
    comp_model.init_opts(max_lr = max_lr, pct_start=0.10)
        
    base_model.fit(crit, dls, num_epochs=num_epochs, stop_on_plateau = early_stop, stop_percent = stop_pct)
    base_model.splice(splice, max_lr=max_lr)
    base_model.fit(crit, dls, num_epochs=5, stop_on_plateau = False)
    base_model.init_opts(pct_start = 0.10, max_lr=max_lr/10)
    base_model.fit(crit, dls, num_epochs=num_epochs, stop_on_plateau=early_stop, stop_percent=stop_pct)
        
    comp_model.fit(crit, dls, num_epochs=num_epochs+5, stop_on_plateau=early_stop, stop_percent=stop_pct)
    comp_model.init_opts(pct_start = 0.10, max_lr=max_lr/10)
    comp_model.fit(crit, dls, num_epochs=num_epochs, stop_on_plateau=early_stop, stop_percent=stop_pct)

    tr_hist, val_hist = base_model.process_history()
    comp_tr_hist, comp_val_hist = comp_model.process_history()

    tr_hist = pd.merge(tr_hist, comp_tr_hist, left_on='iteration', right_on='iteration', how='outer')
    val_hist = pd.merge(val_hist, comp_val_hist, left_on='iteration', right_on='iteration', how='outer')
    
    tr_hist['trial'] = trial
    val_hist['trial'] = trial
    tr_histories.append(tr_hist)
    val_histories.append(val_hist)

    if save: 
        torch.save(base_model, exp_path/'base_model_{}.pkl'.format(trial))
        torch.save(comp_model, exp_path/'comp_model_{}.pkl'.format(trial))

tr_hist = pd.concat(tr_histories).reset_index(drop=True)
val_hist = pd.concat(val_histories).reset_index(drop=True)

tr_hist.to_csv(exp_path/'train_history.csv')
val_hist.to_csv(exp_path/'val_history.csv')

core.plot_loss_comparison(tr_hist, val_hist, save=exp_path/'loss_history.png')

    













