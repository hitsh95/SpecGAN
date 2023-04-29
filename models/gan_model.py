import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
from .base_model import BaseModel
from . import networks
from utils.util import print_network

class GANModel(BaseModel):
    def name(self):
        return 'SpecGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        self.opt = opt

        self.spec_w = opt.spec_width
        self.avg_curves = self.Tensor(nb, 1, self.spec_w) 

        skip = True if opt.skip > 0 else False
        self.netG = networks.define_G(opt)

        if self.isTrain:
            self.netD = networks.define_D(self.spec_w, opt, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        print_network(self.netG)
        if self.isTrain:
            print_network(self.netD)
        if opt.isTrain:
            self.netG.train()
        else:
            self.netG.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        self.fit_curves_A = input['fit_curves'] 
        self.avg_curves_A = input['avg_curve']  
        self.lambda_value = input['lambda']
        self.clsName = input['clsName']

    def test(self):
        self.netG.eval()
        self.real_A = Variable(self.fit_curves_A.float())

        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG.forward(self.real_A)
        else:
           self.fake_B = self.netG.forward(self.real_A)

        self.real_B = self.avg_curves_A
        # self.real_B = torch.zeros_like(self.fake_B).cuda().float()
        # for b in range(self.fake_B.shape[0]):  
        #     self.real_B[b,:self.particle_num[b]] = self.avg_curves_A[b].repeat((self.particle_num[b],1)).cuda().float()
        
        # l2_loss = nn.MSELoss(reduction='sum')
        # n = torch.sum(self.particle_num)
        # loss_spec = l2_loss(self.fake_B, self.real_B) / n

        l2_loss = nn.MSELoss()
        loss_S_L2 = loss_spec.item()
        loss_spec = l2_loss(self.fake_B, self.real_B)

        fake_B = self.fake_B[0].cpu().detach().float().numpy()
        fit_curves_A = self.fit_curves_A[0].cpu().detach().float().numpy()
        avg_curve_A = self.avg_curves_A[0].cpu().float().numpy()

        # return OrderedDict([('real_img', real_img), ('fit_curves_A', fit_curves_A), ('l2_loss', loss_S_L2),
        #                         ('fit_curves_B', fake_B), ('avg_curve_A', avg_curve_A)])
        return OrderedDict([('fit_curves_A', fit_curves_A), ('l2_loss', loss_S_L2),
                                ('fit_curves_B', fake_B), ('avg_curve_A', avg_curve_A)])

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real) # b*128*1
        pred_fake = netD.forward(fake.detach()) # b*128*1
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

   
    def backward_D(self):
        fake_curve = self.fake_B
        real_curve = self.avg_curves_A
        self.loss_D = self.backward_D_basic(self.netD, real_curve, fake_curve, True)
        self.loss_D.backward()

    def forward(self):

        self.real_A = Variable(self.fit_curves_A.float())

        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise

        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG.forward(self.real_A)
        else:
            self.fake_B = self.netG.forward(self.real_A)

    def backward_G(self):
        self.loss_G = 0
        self.loss_spec = 0
     
        self.real_B = self.avg_curves_A.cuda()
        l2_loss = nn.MSELoss()
        self.loss_spec = l2_loss(self.fake_B, self.real_B)

        self.loss_S_L2 = self.loss_spec * self.opt.loss_spec_w
        self.loss_G += self.loss_S_L2

        # GAN loss
        self.loss_G_S = 0
        pred_fake = self.netD.forward(self.fake_B) # b*128*41 -> b*128*1
        if self.opt.use_wgan:
            self.loss_G_S = -pred_fake.mean()
        elif self.opt.use_ragan:
            pred_real = self.netD.forward(self.real_B)
            self.loss_G_S = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                                    self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
        else:
            self.loss_G_S = self.criterionGAN(pred_fake, True)
        self.loss_G += self.loss_G_S

        # tv loss
        self.loss_tv = 0 
        if self.opt.use_tv:
            tv_loss = networks.TVLoss1D(self.opt.loss_tv_w, self.opt.batchSize)
            self.loss_tv = tv_loss(self.fake_B)
        # self.loss_G += self.loss_tv
        
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        if self.opt.phase == 'train' and not self.netG.training:
            self.netG.train()
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step() 

    def get_current_errors(self, epoch):
        G_Total = self.loss_G.item()
        D_S = self.loss_D.item()
        G_S = self.loss_G_S.item()
        S_L2 = self.loss_S_L2.item()
        tv = self.loss_tv.item()

        return OrderedDict([("G_Total", G_Total), ("G_S", G_S), 
            ("S_L2", S_L2), ("D_S", D_S)])
        
    def get_current_visuals(self):
        fake_B = self.fake_B[0].cpu().detach().float().numpy()
        fit_curves_A = self.fit_curves_A[0].cpu().detach().float().numpy()
        avg_curve_A = self.avg_curves_A[0].cpu().float().numpy()
        lambda_value = self.lambda_value[0].cpu().float().numpy()
        

        return OrderedDict([('lambda',lambda_value), ('fit_curves_A', fit_curves_A), ('fit_curves_B', fake_B), 
        ('avg_curve_A', avg_curve_A), ("clsName", self.clsName)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
