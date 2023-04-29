import os.path
from data.base_dataset import BaseDataset
import numpy as np

import pandas as pd
import csv

from sklearn.preprocessing import minmax_scale

class SpecDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_A_train = os.path.join(opt.dataroot, 'lipid_spec_train_res.csv')
        self.dir_A_val = os.path.join(opt.dataroot, 'lipid_spec_val_res.csv')
        self.dir_A_test = os.path.join(opt.dataroot, 'resnet_lipid_outputs.csv')
        self.dir_B = os.path.join(opt.dataroot, 'gt.csv')

        self.pd_reader_B = pd.read_csv(self.dir_B)

        if opt.phase == 'train':
            self.pd_reader_A = pd.read_csv(self.dir_A_train)
        elif opt.phase == 'val':
            self.pd_reader_A = pd.read_csv(self.dir_A_val)
        elif opt.phase == 'test':
            self.pd_reader_A = pd.read_csv(self.dir_A_test)
        else:
            print("opt.phase mode is wrong! please check it!! it should be 'train', 'val' or 'test'. ")
        
        self.A_size = len(self.pd_reader_A)

    def __getitem__(self, index):

        pd_reader = self.pd_reader_A
        i = index

        particle_info = {}
        headers = list(pd_reader.columns.values)
        spec_idx = headers.index('560')
        for k in range(spec_idx):
            key = headers[k]
            particle_info[key] = pd_reader[key][i]

        particle_info['lambda'] = np.array(pd_reader.columns[spec_idx:].values).astype(np.float32)
        
        v = np.array(pd_reader.iloc[i][spec_idx:].values)
        v = minmax_scale(v)

        particle_info['fit_curves'] = np.expand_dims(v,axis=0).astype(np.float32)

        label_list = [x for x in self.pd_reader_B['cls']]
        idx = label_list.index(particle_info['cls'])
        avg_cruve = np.array(self.pd_reader_B.iloc[idx][3:].values)
        self.avg_curve = np.expand_dims(avg_cruve,axis=0).astype(np.float32)

        # data = self.train_A if self.opt.phase == 'train' else self.test_A
        # particle_info = data[index]
        particle_info['avg_curve']=self.avg_curve

        return particle_info

    def __len__(self):
        return self.A_size

    def name(self):
        return 'SpecDataset'


