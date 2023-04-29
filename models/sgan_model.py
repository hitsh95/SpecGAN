import numpy as np
import torch
from torch import nn
from utils.util import print_network
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

class sGANModel(BaseModel):
    def name(self):
        return 'sGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.opt = opt

        self.input_nc = opt.input_nc
        self.netG = networks.define_G(opt)
        self.netD = networks.define_D(self.input_nc, opt, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            self.load_network(self.netD, 'D', which_epoch)

        # define loss functions
        if opt.use_wgan:
            self.criterionGAN = networks.DiscLossWGANGP()
        else:
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        self.adv_weight = 1.0
        self.aux_weight = 1.0

        if self.isTrain:
            self.old_lr = opt.lr
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        print('---------- Networks initialized -------------')
        print_network(self.netG)
        if self.isTrain:
            print_network(self.netD)
            self.netG.train()
            self.netD.train()
        else:
            self.netG.eval()
            self.netD.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        self.real_A = input['fit_curves'].to(self.device)
        self.real_B = input['avg_curve'].to(self.device)
        self.lambda_value = input['lambda']
        self.clsName = input['clsName']
        self.label = input['cls'].to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def test(self):
        self.netG.eval()
        self.netD.eval()
        self.forward()

        self.loss_G = self.cal_loss_G(self.netD, self.real_B, self.fake_B, self.label)
        self.loss_D = self.cal_loss_D(self.netD, self.real_B, self.fake_B, self.real_A, self.label)

    def backward_D(self):
        self.loss_D = self.cal_loss_D(self.netD, self.real_B, self.fake_B.detach(), self.real_A, self.label)
        self.loss_D.backward()

    def cal_loss_D(self, netD, realB, fakeB, realA, label):
        # real
        real_gan_logit, real_cls_logit = self.netD.forward(realB) # b*2  b*num_class
        _,real_cls_logit_A = self.netD.forward(realA)
        fake_gan_logit, _ = self.netD.forward(fakeB)

        if self.opt.use_wgan:
            loss_D_real_gan = real_gan_logit.mean()
            loss_D_fake_gan = fake_gan_logit.mean()
            loss_D_gan = loss_D_fake_gan - loss_D_real_gan + self.criterionGAN.calc_gradient_penalty(netD, 
                                                realB.data, fakeB.data)
        elif self.opt.use_ragan:
            loss_D_gan = (self.criterionGAN(real_gan_logit - torch.mean(fake_gan_logit), True) +
                                      self.criterionGAN(fake_gan_logit - torch.mean(real_gan_logit), False)) / 2
        else:
            loss_D_real_gan = self.criterionGAN(real_gan_logit, True)
            loss_D_fake_gan = self.criterionGAN(fake_gan_logit, False)
            loss_D_gan = (loss_D_real_gan + loss_D_fake_gan) * 0.5

        self.predict_cls_A = torch.max(real_cls_logit_A, dim=1)[1]
        
        real_logit = torch.gather(real_cls_logit, dim=1, index=torch.unsqueeze(label,dim=1))
        real_logit_A = torch.gather(real_cls_logit_A, dim=1, index=torch.unsqueeze(label,dim=1))

        loss_D_real_cls = self.MSE_loss(real_logit, torch.ones_like(real_logit))
        loss_D_real_cls_A = self.MSE_loss(real_logit_A, torch.ones_like(real_logit_A))

        self.loss_D_cls = loss_D_real_cls + loss_D_real_cls_A
        loss_D = self.adv_weight * loss_D_gan + self.aux_weight * self.loss_D_cls
        # loss_D = loss_D_gan
        return loss_D

    def backward_G(self):
        self.loss_G = self.cal_loss_G(self.netD, self.real_B, self.fake_B, self.label)
        self.loss_G.backward()

    def cal_loss_G(self, netD, real_B, fake_B, label):
        loss_G = 0
        self.loss_spec = 0
        data_idx = label != 11

        # MSE loss
        smooth_l1_loss = nn.SmoothL1Loss().to(self.device)
        l1_loss = nn.L1Loss().to(self.device)
        l2_loss = nn.MSELoss().to(self.device)
        self.loss_spec = l2_loss(fake_B[data_idx,:,:], real_B[data_idx,:,:])
        self.loss_L2 = self.loss_spec * self.opt.loss_spec_w
        loss_G += self.loss_L2

        # GAN loss
        real_gan_logit, real_cls_logit = netD.forward(real_B) # b*2  b*num_class
        fake_gan_logit, fake_cls_logit = netD.forward(fake_B)
        if self.opt.use_wgan:
            self.loss_gan = -fake_gan_logit[data_idx,:].mean()
        elif self.opt.use_ragan:
            self.loss_gan = (self.criterionGAN(real_gan_logit[data_idx,:] - torch.mean(real_gan_logit[data_idx,:]), False) +
                                    self.criterionGAN(fake_gan_logit[data_idx,:] - torch.mean(fake_gan_logit[data_idx,:]), True)) / 2
        else:
            self.loss_gan = self.criterionGAN(fake_gan_logit[data_idx,:], True)
        loss_G += self.loss_gan * self.opt.loss_gan_w

        ## auxilary loss
        fake_logit = torch.gather(fake_cls_logit, dim=1, index=torch.unsqueeze(label,dim=1))
        self.loss_G_cls = self.MSE_loss(fake_logit, torch.ones_like(fake_logit))

        # loss_G += self.loss_G_cls

        # tv loss
        self.loss_tv = 0 
        if self.opt.use_tv:
            tv_loss = networks.TVLoss1D(self.opt.loss_tv_w, self.opt.batchSize)
            self.loss_tv = tv_loss(fake_B)
        loss_G += self.loss_tv * self.opt.loss_tv_w

        return loss_G

    def forward(self):
        if self.opt.noise > 0:
            self.noise = torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.)
            self.real_A = self.real_A + self.noise

        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG.forward(self.real_A)
        else:
            self.fake_B = self.netG.forward(self.real_A)

    def optimize_parameters(self):
        # forward
        if self.opt.phase == 'train' and not self.netG.training:
            self.netG.train()
            self.netD.train()
        self.forward()

        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step() 

    def get_current_errors(self, epoch):
        loss_d = self.loss_D.item()
        loss_g = self.loss_G.item()
        loss_aux_D = self.loss_D_cls.item()  # real_A and real_B
        loss_aux_G = self.loss_G_cls.item() # fake B
        loss_tv = self.loss_tv.item()     

        loss_total = loss_d + loss_g

        return OrderedDict([("loss_total", loss_total), ("loss_d", loss_d), 
            ("loss_g", loss_g), ("loss_aux_D", loss_aux_D), ("loss_aux_G", loss_aux_G),
            ("loss_tv", loss_tv)])
        
    def get_current_visuals(self):
        fake_B = self.fake_B.cpu().detach().float().numpy()
        real_A = self.real_A.cpu().detach().float().numpy()
        real_B = self.real_B.cpu().float().numpy()
        lambda_value = self.lambda_value.cpu().float().numpy()
        predict_cls_A = self.predict_cls_A.cpu().int().numpy()
        
        return OrderedDict([('lambda',lambda_value), ('real_A', real_A), ('fake_B', fake_B), 
        ('real_B', real_B), ("clsName", self.clsName), ("predict_cls_A", predict_cls_A)])

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
