import numpy as np
import torch
from torch import nn
from utils.util import print_network
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
from .base_model import BaseModel
from . import networks
import sys


class VAEModel(BaseModel):
    def name(self):
        return 'VAEModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.A_img = self.Tensor(nb, opt.input_nc, size, size)

        self.spec_w = opt.spec_width
        self.fit_curves_B = self.Tensor(nb, 1, self.spec_w) 
        self.avg_curves = self.Tensor(nb, 1, self.spec_w) 

        skip = True if opt.skip > 0 else False
        self.netG = networks.define_G(opt)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            

        if self.isTrain:
            self.old_lr = opt.lr # 0.001
            # define loss functions
            self.vae_loss = networks.VAELoss(opt)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G,'min',patience=30,factor=0.5,min_lr=1e-12,verbose=True)
        
        print('---------- Networks initialized -------------')
        print_network(self.netG)

        if opt.isTrain:
            self.netG.train()
        else:
            self.netG.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        A_img = input['A']
        self.A_img.resize_(A_img.size()).copy_(A_img)
        self.fit_curves_A = input['fit_curves']
        self.fit_boxes = input['fit_boxes']
        self.avg_curves_A = input['avg_curves']
        self.particle_num = input['particle_num']
        self.centers = input['centers']
        self.data_mean = torch.mean(self.fit_curves_A), 
        self.data_std = torch.std(self.fit_curves_A )

    def test(self):
        self.netG.eval()
        self.real_A = Variable(self.fit_curves_A.float())

        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A, self.mu, self.logvar = self.netG.forward(self.real_A)
        else:
           self.fake_B, self.mu, self.logvar = self.netG.forward(self.real_A)

        self.real_B = torch.zeros_like(self.fake_B).cuda().float()
        for b in range(self.fake_B.shape[0]):  
            self.real_B[b,:self.particle_num[b]] = self.avg_curves_A[b].repeat((self.particle_num[b],1)).cuda().float()
        
        l2_loss = nn.MSELoss(reduction='sum')
        n = torch.sum(self.particle_num)
        loss_spec = l2_loss(self.fake_B, self.real_B) / n
        loss_S_L2 = loss_spec.item()

        real_img = util.tensor2imrec(self.A_img.data, self.fit_boxes)
        fake_B = self.fake_B[0].cpu().detach().float().numpy()
        fit_curves_A = self.fit_curves_A[0].cpu().detach().float().numpy()
        avg_curve_A = self.avg_curves_A[0].cpu().float().numpy()

        return OrderedDict([('real_img', real_img), ('fit_curves_A', fit_curves_A), ('l2_loss', loss_S_L2),
                                ('fit_curves_B', fake_B), ('avg_curve_A', avg_curve_A)])

    def forward(self):

        self.real_A = Variable(self.fit_curves_A.float())

        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise

        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A, self.mu, self.logvar = self.netG.forward(self.real_A)
        else:
            self.fake_B, self.mu, self.logvar = self.netG.forward(self.real_A)
        
        
    def backward_G(self, epoch):
        self.loss_G = 0
        self.loss_spec = 0
        
        self.real_B = torch.zeros_like(self.fake_B).cuda().float()
        for b in range(self.fake_B.shape[0]):  
            self.real_B[b,:self.particle_num[b]] = self.avg_curves_A[b].repeat((self.particle_num[b],1)).cuda().float()
        
        n = torch.sum(self.particle_num)
        reconstruction_loss, kl_loss = self.vae_loss(self.fake_B, self.real_B, 
                self.mu, self.logvar, n, self.data_mean, self.data_std, None)
        kl_weight = self.vae_loss.get_kl_weight(epoch)
        self.loss_G = reconstruction_loss + kl_weight*kl_loss

        # tv loss
        self.loss_tv = 0 
        if self.opt.use_tv:
            tv_loss = networks.TVLoss1D(self.opt.loss_tv_w, n)
            self.loss_tv = tv_loss(self.fake_B)
        self.loss_G += self.loss_tv
        
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        # forward
        if self.opt.phase == 'train' and not self.netG.training:
            self.netG.train()
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()

    def get_current_errors(self, epoch):
        G_Total = self.loss_G.item()
        tv = self.loss_tv.item()

        return OrderedDict([("G_Total", G_Total), ("tv_loss",tv)])
        
    def get_current_visuals(self):
        real_img = util.tensor2imrec(self.A_img.data, self.fit_boxes, self.centers)
        fake_B = self.fake_B[0].cpu().detach().float().numpy()
        fit_curves_A = self.fit_curves_A[0].cpu().detach().float().numpy()
        avg_curve_A = self.avg_curves_A[0].cpu().float().numpy()
        

        return OrderedDict([('real_img', real_img), ('fit_curves_A', fit_curves_A), 
                        ('fit_curves_B', fake_B), ('avg_curve_A', avg_curve_A)])

    def save(self, label):
        # self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
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
