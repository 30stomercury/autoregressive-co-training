import torch
import os
import numpy as np
from read_scps import get_fbank_scp, get_bxxx, get_bxxx_scp
from torch.utils.data import Sampler

_MAX_SEQ_LEN = 2600

class collate_fn:
    def __call__(self, batch):
        one_batch = list(zip(*batch))
        idx, name, x, y = one_batch

        batch_size = len(x)
        x_length = [len(x[i]) for i in range(batch_size)]
        y_length = [len(y[i]) for i in range(batch_size)]
        
        max_x_length = max(x_length)
        max_y_length = max(y_length)

        x_padded = torch.zeros(batch_size, max_x_length, len(x[0][0]))
        y_padded = torch.zeros(batch_size, max_y_length, len(y[0][0]))

        for i in range(len(x)):
            x_padded[i][:len(x[i])] = x[i] if torch.is_tensor(x[i]) else torch.Tensor(x[i])
            y_padded[i][:len(y[i])] = y[i] if torch.is_tensor(y[i]) else torch.Tensor(y[i])

        mask = x_padded.eq(0.0)[..., :1]

        return idx, name, x_padded, torch.IntTensor(x_length), y_padded, torch.IntTensor(y_length), mask


class collate_align_fn:
    def __call__(self, batch):
        one_batch = list(zip(*batch))
        idx, name, x, y = one_batch

        batch_size = len(x)
        x_length = [len(x[i]) for i in range(batch_size)]
        y_length = [len(y[i]) for i in range(batch_size)]
        
        max_x_length = max(x_length)
        max_y_length = max(y_length)

        x_padded = torch.zeros(batch_size, max_x_length, len(x[0][0]))
        y_padded = torch.zeros(batch_size, max_y_length)

        for i in range(len(x)):
            x_padded[i][:len(x[i])] = x[i] if torch.is_tensor(x[i]) else torch.Tensor(x[i])
            y_padded[i][:len(y[i])] = y[i] if torch.is_tensor(y[i]) else torch.Tensor(y[i])

        mask = x_padded.eq(0.0)[..., :1]

        return idx, name, x_padded, torch.IntTensor(x_length), y_padded, torch.IntTensor(y_length), mask


class ls_data(torch.utils.data.Dataset):
    def __init__(self, config, part):

        self.config = config
        self.part = part

        self.t_shift = config['t_shift']
        self.num = config['num']
        self.norm = config['norm']
        self.context = config['context']

        info = open(config['mean_var'], 'r').readlines()
        sum_x = np.array([float(x) for x in info[0][1:-2].split(',')])
        sum_x2 = np.array([float(x) for x in info[1][1:-2].split(',')])
        n_frame = int(info[2])
        self.g_mean = sum_x / n_frame
        self.g_var = sum_x2 / n_frame - self.g_mean**2

        # fbank
        self.fbanks = get_fbank_scp(config['set'])

    def __len__(self):
        return len(self.fbanks.scps) if self.num == 'all' else self.num

    def __getitem__(self, index):
        name, mat = self.fbanks[index]
        if self.norm:
            mat = (mat - self.g_mean) / (self.g_var ** 0.5)
        mat = np.array(mat)[:_MAX_SEQ_LEN]

        if self.context > 1:
            l_context = (self.context - 1) // 2
            r_context = self.context - 1 - l_context
            pad_mat = np.pad(mat, ((l_context, r_context), (0, 0)), mode='reflect')
            long_mat = np.concatenate([pad_mat[i: i+len(mat)] for i in range(self.context)], axis=-1)
            mat = long_mat

        return index, name, mat[:-self.t_shift], mat[self.t_shift:]


class wsj_data(torch.utils.data.Dataset):
    def __init__(self, config, part='si284-0.9-train'):

        self.config = config
        self.part = part

        self.num = config['num']
        self.norm = config['norm']
        self.context = config['context']

        info = open(config['mean_var'], 'r').readlines()
        sum_x = np.array([float(x) for x in info[0][1:-2].split(',')])
        sum_x2 = np.array([float(x) for x in info[1][1:-2].split(',')])
        n_frame = int(info[2])
        self.g_mean = sum_x / n_frame
        self.g_var = sum_x2 / n_frame - self.g_mean**2

        # Get data
        self.fbanks = get_fbank_scp(config['set'].format(part))
        if part in ['dev93', 'eval92']:
            self.target = get_bxxx(config['target'].format(part), config['tokens'])
        else:
            self.target = get_bxxx_scp(config['target'].format(part), config['tokens'])

    def __len__(self):
        return len(self.fbanks.scps) if self.num == 'all' else self.num

    def __getitem__(self, index):
        name, mat = self.fbanks[index]
        if self.norm:
            mat = (mat - self.g_mean) / (self.g_var ** 0.5)
        mat = np.array(mat)

        if self.context > 1:
            l_context = (self.context - 1) // 2
            r_context = self.context - 1 - l_context
            pad_mat = np.pad(mat, ((l_context, r_context), (0, 0)), mode='reflect')
            long_mat = np.concatenate([pad_mat[i: i+len(mat)] for i in range(self.context)], axis=-1)
            mat = long_mat

        # Get phns
        y = np.array(self.target[name], dtype=np.float32)
         
        return index, name, mat, y
