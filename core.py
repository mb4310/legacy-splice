import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import copy; import time; import pdb
import torch.nn as nn
import torch
import fastai


class Smoother():
    def __init__(self, beta=0.95):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.vals = []

    def add_value(self, val):
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1-self.beta)*val
        self.vals.append(self.mov_avg/(1-self.beta**self.n))

    def process(self,array):
        for item in array:
            self.add_value(item)
        return self.vals

    def reset(self):
        self.n, self.mov_avg, self.vals = 0,0,[]

class Stepper():
    def __init__(self, opt):
        self.it = 0
        self.opt = opt
        self.nits = 1

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()
    
    @staticmethod
    def cosine_anneal(pct, max_val, min_val):
        return min_val + (max_val - min_val) / 2 *(1+np.cos(np.pi * pct))
    
    @staticmethod
    def exp_anneal(pct, start, stop):
        return start * (stop/start)**pct
    
    @staticmethod
    def linear_anneal(pct, start, stop):
        return (1-pct)*start + pct*stop
    
class OneCycle(Stepper):
    def __init__(self, opt, nits=1, max_lr=1e-3, momentums=[0.85,0.95], div=25, pct_start=0.3):
        super(OneCycle, self).__init__(opt)
        self.nits = nits
        self.max_lr = max_lr
        self.momentums = momentums
        self.div = div
        self.pct_start = pct_start
        self.phase = 0
        self.switch = int(pct_start * nits)
    
    def step(self):
        self.opt.step()
        self.it += 1
        if self.phase == 0: 
            pct = self.it / (self.nits * self.pct_start)
            new_lr = self.cosine_anneal(pct, self.max_lr/self.div, self.max_lr)
            new_mom = self.cosine_anneal(pct, self.momentums[1], self.momentums[0])
            for group in self.opt.param_groups:
                group['lr'] = new_lr
                if 'betas' in group.keys():
                    group['betas'] = (new_mom, group['betas'][1])
                else:
                    group['momentum'] = new_mom
            if self.it > self.switch:
                self.phase += 1
                self.it = 0
        
        else: 
            pct = self.it / (self.nits * (1-self.pct_start))
            new_lr = self.cosine_anneal(pct, self.max_lr, self.max_lr * 1e-4)
            new_mom = self.cosine_anneal(pct, self.momentums[0], self.momentums[1])
            for group in self.opt.param_groups:
                group['lr'] = new_lr
                if 'betas' in group.keys():
                    group['betas'] = (new_mom, group['betas'][1])
                else:
                    group['momentum'] = new_mom

class LearningRateFinder(Stepper):
    def __init__(self, opt, nits=1, min_lr=1e-6, max_lr=1e1):
        super(LearningRateFinder, self).__init__(opt)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.pct_start = 0
        self.nits = nits
        for group in self.opt.param_groups:
            group['lr'] = min_lr
    
    def step(self):
        self.opt.step()
        self.it+=1 
        new_lr = self.exp_anneal(self.it / self.nits, self.min_lr, self.max_lr)
        for group in self.opt.param_groups:
            group['lr'] = new_lr
    
    @staticmethod
    def plot_lr_find(tr_history, clip=True):                                
        fig, ax = plt.subplots()
        if clip:
            end = int(0.90 * len(tr_history))
            tr_history = tr_history.iloc[:end]
        ax.plot(tr_history.learning_rate, tr_history.tr_loss)
        ax.set_xscale('log')
        ax.legend()
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')            
            
    def lr_find(self, model, tr_dl, criterion):
        tr_losses = []
        lrs = []
        iterator = iter(tr_dl)
        self.it = 0 
        while self.it <= self.nits:
            inputs, labels = next(iterator)
            self.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.step()
            tr_losses.append(loss.item())
            lrs.append(self.opt.param_groups[-1]['lr'])
        tr_losses = Smoother(beta=0.99).process(tr_losses)
        tr_history = pd.DataFrame({'tr_loss':tr_losses, 'learning_rate':lrs})
        self.plot_lr_find(tr_history)
        return None 
    
class UnfreezeAnneal(Stepper):
    def __init__(self, opt, nits=1, max_lr=1e-3, pct_start = 0.3):
        super(UnfreezeAnneal, self).__init__(opt)
        self.nits = nits
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.phase = 0
        self.switch = int(pct_start * nits)
    
    def step(self):
        self.opt.step()
        self.it+=1 
        if self.phase==0:
            pct = self.it / (self.nits * self.pct_start)
            new_lr = self.linear_anneal(pct, 0, self.max_lr * 1e-5)
            for group in self.opt.param_groups:
                group['lr'] = new_lr
            if self.it > self.switch:
                self.phase += 1
                self.it = 0
        else:
            pct = self.it / (self.nits * (1-self.pct_start))
            new_lr = self.cosine_anneal(pct, self.max_lr * 1e-5, self.max_lr)
            for group in self.opt.param_groups:
                group['lr'] = new_lr  
       
class cnn(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.l1 = fastai.layers.conv_layer(ni = 3, nf = 64, ks = 7, stride=2, padding=3)
        self.block1 = fastai.layers.res_block(64)
        self.l2 = fastai.layers.conv_layer(ni=64, nf=128, ks=3, stride=2, padding=1)
        self.block2 = fastai.layers.res_block(128)
        self.l3 = fastai.layers.conv_layer(ni=128,nf=256, ks=3, stride=2, padding=1)
        self.out = fastai.vision.create_head(nf = 512, nc = nc)
        self.layers = nn.Sequential(self.l1, self.block1, self.l2, self.block2, self.l3, self.out)
        self.blocks = OrderedDict({1:self.block1, 2:self.block2})
        self.tr_losses = []
        self.val_losses = []
        self.lrs = OrderedDict()
        self.steppers = []
    
    def init_params(self):
        for module in self.modules():
            if module._get_name() == 'BatchNorm2d':
                self.initialize_bn(module)
            else:
                for param in module.parameters():
                    if param.dim() > 1: 
                        param = nn.init.kaiming_normal_(param)
                    
    @staticmethod        
    def initialize_conv(conv, cuda=True):
        center = conv.kernel_size[0] ** 2 // 2
        data = torch.zeros(conv.kernel_size).put_(torch.tensor([center]), torch.tensor([1.]))
        params = list(conv.parameters())
        param = params[0]
        if param.size(0) != param.size(1):
            raise TypeError('Conv must be "square"... (in_features == out_features)')
        param.data = torch.zeros(param.data.size())
        for k in range(param.size(0)):
            param.data[k,k,:,:] = data
        if cuda:
            param.data = param.data.cuda()
        if len(params) > 1: 
            bias = params[1]
            bias.data = torch.zeros(bias.data.size())
            if cuda:
                bias.data = bias.data.cuda()
        return conv
    
    @staticmethod
    def initialize_bn(bn, cuda=True):
        bn.weight.data.fill_(1)
        bn.bias.data.fill_(0)
        if cuda: bn.cuda()
        return bn            
    
    def initialize_resblock(self,nf=128, **kwargs):
        res = fastai.layers.res_block(nf)
        self.initialize_conv(res[1][0])
        self.initialize_conv(res[0][0])
        self.initialize_bn(res[1][2])
        self.initialize_bn(res[0][2])        
        return res
    
    def splice_steppers(self, max_lr=1e-3, **kwargs):
        opt1 = torch.optim.Adam(self.old_params)
        opt2 = torch.optim.Adam(self.new_params)
        stepper1 = UnfreezeAnneal(opt1, max_lr = max_lr/10, **kwargs)
        stepper2 = OneCycle(opt2, max_lr=max_lr, **kwargs)
        self.steppers = [stepper2, stepper1]
        self.old_params = None
        self.new_params = None
        return None
    
    def splice(self, splice_blocks = [[128,2,0]], **kwargs):
        self.old_params = (x for x in list(self.parameters()))
        self.new_params = []
        for nf, b_idx, i_idx in splice_blocks:
            res = self.initialize_resblock(nf)
            for layer in res[::-1]:
                self.blocks[b_idx].insert(i_idx,layer)
                self.new_params += list(layer.parameters())
        self.new_params = (x for x in self.new_params)
        self.splice_steppers(**kwargs)
        return None
        
    def init_opts(self, max_lr=1e-3, **kwargs):
        opt = torch.optim.Adam(self.parameters())
        self.steppers = [OneCycle(opt, max_lr=max_lr, **kwargs)]
        return None
    
    def forward(self, x):
        return self.layers(x)   
    
    def fit(self, criterion, dataloaders, num_epochs=1, stop_on_plateau=True, stop_percent=0.01):
        start = time.time()
        end_early = False
        dataset_sizes = {'train':len(dataloaders['train'].dataset), 'val':len(dataloaders['val'].dataset)}
        val_loss_buffer = [1e8, 1e8, 1e8, 1e8, 1e8]

        for k,stepper in enumerate(self.steppers):
            stepper.nits = num_epochs * len(dataloaders['train'].dataset) / (dataloaders['train'].batch_size)
            stepper.switch = int(stepper.pct_start * stepper.nits)
            stepper.it = 0
            stepper.phase = 0
            self.lrs[k] = []
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-'*10)

            for phase in ['train','val']:
                if end_early:
                    break
                    
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_losses = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    for stepper in self.steppers:
                        stepper.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        for k, stepper in enumerate(self.steppers):
                            stepper.step()
                            self.lrs[k].append(stepper.opt.param_groups[-1]['lr'])
                        self.tr_losses.append(loss.item())
                    
                    else:
                        self.val_losses.append(loss.item())
                    
                    running_losses += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_losses / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                
                if phase == 'val':
                    decrease_percent = (epoch_loss - np.mean(val_loss_buffer))/np.min(val_loss_buffer)
                    if (decrease_percent > -stop_percent) and stop_on_plateau:
                        end_early = True
                              
                    val_loss_buffer.pop(0)
                    val_loss_buffer.append(epoch_loss)
                                    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        elapsed_time = time.time() - start
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))    
        print(elapsed_time)
        return None
 
    def reset_history(self):
        self.tr_losses = []
        self.val_losses = []
        
    def process_history(self):
        tr_losses = Smoother().process(self.tr_losses)
        val_losses = Smoother(beta=0.98).process(self.val_losses)
        tr_history = pd.DataFrame({'tr_loss':tr_losses}).reset_index().rename(columns={'index':'iteration'})
        val_history = pd.DataFrame({'val_loss':val_losses}).reset_index().rename(columns={'index':'iteration'})
        return tr_history, val_history
    
    def plot_lr(self):
        lrs = pd.DataFrame(self.lrs)
        l = len(lrs.columns)
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration')
        ax.set_ylabel('learning rate')
        if  l == 2:
            lrs.columns = ['new_parameters','old_parameters']
            lrs['new_parameters'].plot.line()
            lrs['old_parameters'].plot.line()
        else:
            lrs.columns = ['param_group_%s'%(k) for k in lrs.columns]
            for col in lrs.columns:
                lrs[col].plot.line()
        ax.legend()
        return None
    
    def plot_history(self, save=None):
        x,y = self.process_history()
        fig, ax = plt.subplots(2,1, figsize=(16,8))
        x.tr_loss.plot.line(ax=ax[0], color='y')
        y.val_loss.plot.line(ax=ax[1], color='r')
        for j in ax:
            j.set_xlabel('Iteration')
            j.set_ylabel('Loss')
            j.legend()
            
        if save is not None:
            plt.savefig(save)
            
        return None
            
def plot_loss_comparison(x,y, save=None):
    x.columns = ['iteration', 'spliced_model', 'unspliced_model', 'trial']
    y.columns = ['iteration', 'spliced_model', 'unspliced_model', 'trial']

    fig, ax = plt.subplots(2,1, figsize=(16,8))

    x_means = x.groupby('iteration')['spliced_model','unspliced_model'].mean().reset_index()
    y_means = y.groupby('iteration')['spliced_model','unspliced_model'].mean().reset_index()

    x_means.spliced_model.plot.line(ax=ax[0], color='y')
    x_means.unspliced_model.plot.line(ax=ax[0], color='r')
    y_means.spliced_model.plot.line(ax=ax[1], color='y')
    y_means.unspliced_model.plot.line(ax=ax[1], color='r')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Training Loss')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Validation Loss')
    ax[0].legend()
    ax[1].legend()

    for trial in x.trial.unique():
        rx = x[x.trial == trial].reset_index(drop=True)
        rx.spliced_model.plot.line(ax=ax[0], color='y', alpha=0.2)
        rx.unspliced_model.plot.line(ax=ax[0], color='r', alpha=0.2)

        ry = y[y.trial == trial].reset_index(drop=True)
        ry.spliced_model.plot.line(ax=ax[1], color='y', alpha=0.2)
        ry.unspliced_model.plot.line(ax=ax[1], color='r', alpha=0.2)
    
    if save is not None:
        plt.savefig(save)
        
    return None
        
        
        
 