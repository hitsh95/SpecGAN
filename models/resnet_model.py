import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
import numpy as np
from utils.util import print_network

class ResnetModel(BaseModel):
    def name(self):
        return 'ResnetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.A_img = self.Tensor(nb, opt.input_nc, size, size)

        self.spec_w = opt.spec_width
        self.fit_curves_B = self.Tensor(nb, 1, self.spec_w) 
        self.avg_curves = self.Tensor(nb, 1, self.spec_w) 

        self.netG = networks.define_G(opt)
        # define loss functions
        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.BCEWithLogitsLoss()

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
      
        print('---------- Networks initialized -------------')
        print_network(self.netG)
        if opt.isTrain:
            self.netG.train()
        else:
            self.netG.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        device = torch.device(self.opt.gpu_ids[0] if self.opt.gpu_ids else "cpu")

        self.fit_curves_A = input['fit_curves'] 
        self.avg_curves_A = input['avg_curve']  
        self.lambda_value = input['lambda']
        self.label = input['cls']
        self.clsName = input['clsName']

        self.label = self.label.to(device)

    def forward(self):
        self.real_A = self.fit_curves_A.float()
        self.logits = self.netG(self.real_A)

    def test(self):
        self.netG.eval()
        self.real_A = self.fit_curves_A.float()
        logits = self.netG(self.real_A)
        self.total_loss = self.loss_function(logits.squeeze(dim=-1), self.label.float()).item()

        self.predict_y = torch.round(torch.sigmoid(logits)).squeeze(dim=-1)
        # predict_y = torch.max(logits, dim=1)[1]
        label = self.label.detach().cpu().float().numpy()
        
        # predict_y = predict_y.detach().cpu().float().numpy()
        idx = torch.eq(self.predict_y, self.label)

        self.labels_num,self.labels_pred = [], []
        for c in range(self.opt.num_class):
            i = np.argwhere(label == c)[:,0]
            acc_i = idx[i].sum().cpu().numpy()
            total_i = i.shape[0]

            self.labels_num.append(total_i)
            self.labels_pred.append(acc_i)

        # self.acc_n = idx.sum().cpu().numpy()
        
        return OrderedDict([('labels_num', self.labels_num), ('labels_pred', self.labels_pred)])
        
    def backward_G(self):
        self.total_loss = self.loss_function(self.logits.squeeze(dim=-1), self.label.float()) 
        self.total_loss.backward()

    def optimize_parameters(self):
        # forward
        if self.opt.phase == 'train' and not self.netG.training:
            self.netG.train()
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self, epoch):
        total_loss = self.total_loss if isinstance(self.total_loss, float) else self.total_loss.item()
        return OrderedDict([("total_loss", total_loss)])
        
    def get_current_visuals(self):
        # real_img = util.tensor2imrec(self.A_img.data, self.fit_boxes, self.centers)
        fit_curves_A = self.fit_curves_A.cpu().detach().float().numpy()
        avg_curve_A = self.avg_curves_A.cpu().float().numpy()
        lambda_value = self.lambda_value.cpu().float().numpy()
        predict_cls_A = self.predict_y.cpu().int().numpy()
        
        return OrderedDict([('lambda',lambda_value), ('real_A', fit_curves_A), 
        ('real_B', avg_curve_A), ('clsName', self.clsName), ("predict_cls_A", predict_cls_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self):
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
