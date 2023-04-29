import torch
import torch.nn as nn
from torch.nn import init

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# from torch.utils.serialization import load_lua

from utils.util import get_norm_layer, weights_init
###############################################################################
# Functions
###############################################################################

def define_G(opt):
    netG = None
    skip = opt.skip
    gpu_ids = opt.gpu_ids

    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=opt.norm)
    
    if use_gpu:
        assert(torch.cuda.is_available())

    if opt.model == 'resnet':
        include_top = opt.include_top
        netG = Resnet([2,2,2,2], num_classes=opt.num_class, include_top=include_top) # resnet18
    elif opt.which_model_netG == 'Unet_Gen':
        netG = UnetGenerator(opt, skip, norm_layer=norm_layer)
    elif opt.which_model_netG == 'Vae_Gen':
        netG = VAEGenerator(opt, skip)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % opt.which_model_netG)
    if len(gpu_ids) >= 0:
        netG.cuda(device=gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG

def define_D(input_nc, opt, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=opt.norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if opt.which_model_netD == 'basic':
        netD = NLayerDiscriminator(
            input_nc, opt.ndf, opt.n_layers_D, norm_layer=norm_layer, use_sigmoid=opt.no_lsgan, gpu_ids=gpu_ids)
    elif opt.which_model_netD == 'MLP':
        netD = MLPDiscriminator(
            input_nc, use_sigmoid=opt.no_lsgan, gpu_ids=gpu_ids)
    elif opt.which_model_netD =='AuxilaryD': # Discriminator with auxiliary
        netD = AuxilaryDiscriminator(
            input_nc, opt.spec_width, opt.num_class, opt.n_layers_D, opt.ndf)
            
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  opt.which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD

##############################################################################
# Classes
##############################################################################

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var  # b*128*1
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class DiscLossWGANGP():
    def __init__(self):
        self.LAMBDA = 10

    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10

    # def get_g_loss(self, net, realA, fakeB):
    #     # First, G(A) should fake the discriminator
    #     self.D_fake = net.forward(fakeB)
    #     return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(
                                            disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                            ** 2).mean() * self.LAMBDA
        return gradient_penalty

class TVLoss1D(nn.Module):
    def __init__(self, TVLoss_weight=1, count=1):
        super(TVLoss1D, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.count = count

    def forward(self, x):
        w_x = x.size()[2]
        w_tv = torch.pow((x[:, :, 1:]-x[:, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*(w_tv/self.count)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class Resnet(nn.Module):
    def __init__(self, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64, spec_width=120):
        super(Resnet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.width_per_group = width_per_group

        self.Linear1 = nn.Linear(spec_width, 4096) # b*spec_width->b*4096

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResnetBlock, 64, blocks_num[0])
        self.layer2 = self._make_layer(ResnetBlock, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(ResnetBlock, 256, blocks_num[2], stride=2)  
        self.layer4 = self._make_layer(ResnetBlock, 512, blocks_num[3], stride=2)  
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            self.fc = nn.Linear(512 * ResnetBlock.expansion, num_classes)

        for m in self.modules():  
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(  
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))  

        layers = [] 
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
       
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)  

    def forward(self, x):

        b,c,_ = x.shape # [b, 1, spec_width]
        x = self.Linear1(x).unsqueeze(-1).view(b, c, 64, 64)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  

        if self.include_top:
            x = self.avgpool(x)  
            x = torch.flatten(x, 1) 
            x = self.fc(x)  
        
        # Softmax = nn.Softmax(dim=1) 
        # x=Softmax(x)

        return x

# Define a resnet block
class ResnetBlock(nn.Module):
    expansion = 1  
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  

    def forward(self, x):
        identity = x  
        if self.downsample is not None:  
            identity = self.downsample(x)  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  

        out += identity  
        out = self.relu(out)
        return out

class UnetGenerator(nn.Module):
    def __init__(self, opt, skip, norm_layer=nn.BatchNorm2d):
        super(UnetGenerator, self).__init__()
        self.opt = opt
        self.skip = skip
        p = 1
        self.Linear1 = nn.Linear(opt.spec_width, 4096) # b*spec_width->b*4096

        self.conv1_1 = nn.Conv2d(1, 256, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.max_pool1 = nn.AvgPool2d(
            2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.max_pool2 = nn.AvgPool2d(
            2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.max_pool3 = nn.AvgPool2d(
            2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv4 = nn.Conv2d(512, 512, 3, padding=p)
        self.conv5_1 = nn.Conv2d(1024, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 512, 3, padding=p)
        self.conv6_1 = nn.Conv2d(1024, 512, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv6 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv7_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv8 = nn.Conv2d(256, 1, 1)
        self.Linear2 = nn.Linear(4096, opt.spec_width)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        if self.opt.norm in ['batch', 'instance', 'synBN']:
            self.bn1_1 = norm_layer(256)
            self.bn1_2 = norm_layer(256)
            self.bn2_1 = norm_layer(512)
            self.bn2_2 = norm_layer(512)
            self.bn3_1 = norm_layer(512)
            self.bn3_2 = norm_layer(512)
            self.bn4_1 = norm_layer(512)
            self.bn4_2 = norm_layer(512)
            self.bn5_1 = norm_layer(512)
            self.bn5_2 = norm_layer(512)
            self.bn6_1 = norm_layer(512)
            self.bn6_2 = norm_layer(512)
            self.bn7_1 = norm_layer(256)
            self.bn7_2 = norm_layer(256)


    def forward(self, input):
        b,c,_ = input.shape # [b, 1, 120]
        x = self.Linear1(input).unsqueeze(-1).view(b, c, 64, 64)

        NormType = self.opt.norm if self.opt.norm in ['batch', 'instance', 'synBN'] else False

        x = self.LReLU1_1(self.conv1_1(x))
        x = self.bn1_1(x) if NormType else x
        conv1 = self.LReLU1_2(self.conv1_2(x))
        conv1 = self.bn1_2(conv1) if NormType else conv1
        x = self.max_pool1(conv1) # 32*32*256

        x = self.LReLU2_1(self.conv2_1(x))
        x = self.bn2_1(x) if NormType else x
        conv2 = self.LReLU2_2(self.conv2_2(x))
        conv2 = self.bn2_2(conv2) if NormType else conv2
        x = self.max_pool2(conv2)  # 16*16*512
        
        x = self.LReLU3_1(self.conv3_1(x))
        x = self.bn3_1(x) if NormType else x
        conv3 = self.LReLU3_2(self.conv3_2(x))
        conv3 = self.bn3_2(conv3) if NormType else conv3
        x = self.max_pool3(conv3)  # 8*8*512

        x = self.LReLU4_1(self.conv4_1(x))
        x = self.bn4_1(x) if NormType else x
        conv4 = self.LReLU4_2(self.conv4_2(x))
        conv4 = self.bn4_2(conv4) if NormType else conv4 # 8

        conv4 = F.interpolate(conv4, scale_factor=2,
                                mode='bilinear', align_corners=False) # 16
        up5 = torch.cat([self.deconv4(conv4), conv3], 1)
        x = self.LReLU5_1(self.conv5_1(up5))
        x = self.bn5_1(x) if NormType else x
        conv5 = self.LReLU5_2(self.conv5_2(x))
        conv5 = self.bn5_2(conv5) if NormType else conv5 # 8

        conv5 = F.interpolate(conv5, scale_factor=2,
                                mode='bilinear', align_corners=False) # 32
        up6 = torch.cat([self.deconv5(conv5), conv2], 1)
        x = self.LReLU6_1(self.conv6_1(up6))
        x = self.bn6_1(x) if NormType else x
        conv6 = self.LReLU6_2(self.conv6_2(x))
        conv6 = self.bn6_2(conv6) if NormType else conv6 

        conv6 = F.interpolate(conv6, scale_factor=2,
                                mode='bilinear', align_corners=False)
        up7 = torch.cat([self.deconv6(conv6), conv1], 1)
        x = self.LReLU7_1(self.conv7_1(up7))
        x = self.bn7_1(x) if NormType else x
        conv7 = self.LReLU7_2(self.conv7_2(x))
        conv7 = self.bn7_2(conv7) if NormType else conv7 
        
        latent = self.conv8(conv7)
        latent = latent.view(b,c,4096,-1).squeeze(-1)
        latent = self.Linear2(latent)

        if self.skip:
            latent = self.tanh(latent)
            output = latent + input*self.opt.skip
            output = (output - torch.min(output)) / \
                    (torch.max(output)-torch.min(output))
        else:
            latent = self.sigmoid(latent)
            output = latent

        if self.skip:
            return output, latent
        else:
            return output

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if self.opt.use_norm_d:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=kw, stride=2, padding=padw),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        if self.opt.use_norm_d:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                        kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                        kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)

class MLPDiscriminator(nn.Module):
    def __init__(self, input_nc, use_sigmoid=False, gpu_ids=[], use_dropout=False):
        super(MLPDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        hidden1 = 256   # 第二层节点数
        hidden2 = 64   # 第三层节点数

        sequence = [
            nn.Linear(input_nc, hidden1),
            nn.LeakyReLU(0.2, True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.2)]

        sequence += [
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2, True)
        ]

        if use_dropout:
            sequence += [nn.Dropout(0.2)]

        sequence += [
            nn.Linear(hidden2, 1),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        output = self.model(x)  # input [b,128,120]
        if self.use_sigmoid:
            print("sigmoid")
            output = self.sigmoid(output)

        return output

class AuxilaryDiscriminator(nn.Module):
    def __init__(self, input_nc, spec_width, num_class, n_layers=3, ndf=64):
        super(AuxilaryDiscriminator, self).__init__()
        self.num_class = num_class
       
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)] # 1->64

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)] 

        mult = 2 ** (n_layers - 2 - 1)   # 64->128
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*model)

        # Class Activation Map
        mult = 2 ** (n_layers - 2) 

        self.Linear1 = nn.Linear(spec_width, 4096) # b*spec_width->b*4096
        self.pad = nn.ReflectionPad2d(1)

        self.gan_fc_avg = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False)) # b,128 -> b,1
        self.gan_fc_max = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))

        self.cls_fc_avg = nn.utils.spectral_norm(nn.Linear(ndf * mult, num_class, bias=False))
        self.cls_fc_max = nn.utils.spectral_norm(nn.Linear(ndf * mult, num_class, bias=False))

        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True) #b,256 -> b,128
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        b,c,_ = input.shape # [b, 1, spec_width]
        x0 = self.Linear1(input).unsqueeze(-1).view(b, c, 64, 64) # [B,1,64,64]
        x0 = self.model(x0) # feature map [B,128,31,31]
        
        x = x0
        x_avg = torch.nn.functional.adaptive_avg_pool2d(x, 1) # [B,128,1,1]
        x_max = torch.nn.functional.adaptive_max_pool2d(x, 1)
        
        gan_logit_avg = self.gan_fc_avg(x_avg.view(x.shape[0], -1)).unsqueeze(2) # [B,1,1]
        gan_logit_max = self.gan_fc_max(x_max.view(x.shape[0], -1)).unsqueeze(2)  
        gan_logit = torch.cat([gan_logit_avg, gan_logit_max], 2)  # real or fake #[B,2]

        gan_logit = torch.mean(gan_logit,dim=2,keepdim=False) # [B,num_class]

        # gan_avg_weight = list(self.gan_fc_avg.parameters())[0]
        # gan_max_weight = list(self.gan_fc_max.parameters())[0]
        # x_avg = x * gan_avg_weight.unsqueeze(2).unsqueeze(3)
        # x_max = x * gan_max_weight.unsqueeze(2).unsqueeze(3)

        # x = torch.cat([x_avg, x_max], 1)
        # x = self.leaky_relu(self.conv1x1(x))  
        # x = self.pad(x)

        x = x0
        x_avg_2 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_max_2 = torch.nn.functional.adaptive_max_pool2d(x, 1)

        cls_logit_avg = self.cls_fc_avg(x_avg_2.view(x.shape[0], -1)).unsqueeze(2) # [B,num_class,1]
        cls_logit_max = self.cls_fc_max(x_max_2.view(x.shape[0], -1)).unsqueeze(2)
        cls_logit = torch.mean(torch.cat([cls_logit_avg, cls_logit_max], 2),dim=2,keepdim=False) # [B,num_class]
        # Softmax = nn.Softmax(dim=1) 
        # cls_logit=Softmax(cls_logit)

        return gan_logit, cls_logit

##############################################################################
# VAE
##############################################################################
class Down_conv(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 32, init_filters = 32, n_filters_per_depth=2,
                 kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        A helper Module that performs either 2 or 3 convolutions and 1 MaxPool.
        A ReLU activation follows each convolution.
        """
        super(Down_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        
        ins = self.in_channels
        outs = self.out_channels
        
        self.conv1 = nn.Conv2d(in_channels=ins, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.conv2 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        if(self.n_filters_per_depth==3):
            self.conv3 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)    
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if(self.n_filters_per_depth==3):
            x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x
    
    
class Up_conv(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 32, init_filters = 32, n_filters_per_depth=2, 
                 kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        """
        A helper Module that performs either 2 or 3 convolutions and 1 UpConvolution.
        A ReLU activation follows each convolution.
        """
        super(Up_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters 
        self.n_filters_per_depth = n_filters_per_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        
        ins = self.in_channels
        outs = self.out_channels
        
        self.conv1 = nn.Conv2d(in_channels=ins, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.conv2 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        if(self.n_filters_per_depth==3):
            self.conv3 = nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.convtranspose = nn.ConvTranspose2d(in_channels=outs, out_channels=outs, kernel_size=2, stride=2)    
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if(self.n_filters_per_depth==3):
            x = F.relu(self.conv3(x))
        x = self.convtranspose(x)
        return x


class VAEGenerator(nn.Module):    
    def __init__(self, opt, skip, gaussian_noise_std=None, noise_model=None):
        super(VAEGenerator, self).__init__()
        self.opt = opt
        self.skip = skip
        self.gaussian_noise_std = gaussian_noise_std
        self.noise_model = noise_model
        
        self.z_dim = opt.z_dim # 64
        self.in_channels = 1 # 128
        self.init_filters = opt.init_filters  # 256
        self.n_filters_per_depth = opt.n_filters_per_depth # 2
        self.n_depth = opt.n_depth # 2
        self.spec_w = opt.spec_width

        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.bias = True
        self.kl_annealing = True
        self.groups = 1        
        self.collapse = False
        
        ## Encoder
        self.down_convs = []
        self.Linear1 = nn.Linear(self.spec_w, 4096) # b*128*41
        self.convmu = nn.Conv2d(in_channels=self.init_filters*(2**(self.n_depth-1)), out_channels=self.z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        self.convlogvar = nn.Conv2d(in_channels=self.init_filters*(2**(self.n_depth-1)), out_channels=self.z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups = self.groups)
        
        for i in range(self.n_depth):
            ins = self.in_channels if i==0 else outs
            outs = self.init_filters*(2**i)
            down_conv = Down_conv(in_channels=ins, out_channels = outs, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth)
            self.down_convs.append(down_conv)
     
        self.down_convs = nn.ModuleList(self.down_convs)


        ## Decoder
        self.up_convs = []
        self.convrecon = nn.Conv2d(in_channels=self.init_filters, out_channels=self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, groups=self.groups)
        self.Linear2 = nn.Linear(4096, self.spec_w)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for i in reversed(range(self.n_depth)):
            ins = self.z_dim if i==(self.n_depth-1) else outs
            outs = self.init_filters*(2**i)
            up_conv = Up_conv(in_channels=ins, out_channels = outs, init_filters = self.init_filters, n_filters_per_depth=self.n_filters_per_depth)
            self.up_convs.append(up_conv)
            
        self.up_convs = nn.ModuleList(self.up_convs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(mu)
        z = mu + epsilon*std
        return z

    def forward(self, input):
        self.data_mean, self.data_std = torch.mean(input), torch.std(input)
        x = (input-self.data_mean) / self.data_std
        b,c,_ = x.shape #[b,128,41]

        # Encoder
        x = self.Linear1(x).unsqueeze(-1).view(b, c, 64, 64)
        for i, module in enumerate(self.down_convs):
            x = module(x)
        
        mu = self.convmu(x) # [b,64,16,16]
        logvar = self.convlogvar(x) # [b,64,16,16]

        # sample
        x = self.reparameterize(mu, logvar)

        # Decoder
        for i, module in enumerate(self.up_convs):
            x = module(x)

        recon = (self.convrecon(x)* self.data_std)+self.data_mean 
        latent = recon.view(b,c,4096,-1).squeeze(-1)
        latent = self.Linear2(latent)

        if self.skip:
            latent = self.tanh(latent)
            output = latent + input*self.opt.skip
            output = (output - torch.min(output)) / \
                    (torch.max(output)-torch.min(output))
            return output, latent, mu, logvar
        else:
            latent = self.sigmoid(latent)
            output = latent
            return output, mu, logvar


class VAELoss(nn.Module):
    def __init__(self, opt):
        super(VAELoss, self).__init__()
        self.kl_annealing = opt.kl_annealing
        self.kl_start = opt.kl_start
        self.kl_annealtime = opt.kl_annealtime


    def get_kl_weight(self, epoch):
        if(self.kl_annealing==True):
            #calculate weight
            kl_weight = (epoch-self.kl_start) * (1.0/self.kl_annealtime)
            # clamp to [0,1]
            kl_weight = min(max(0.0,kl_weight),1.0)
        else:
            kl_weight = 1.0 
        return kl_weight


    def lossFunctionKLD(self, mu, logvar):
        """Compute KL divergence loss. 
        Parameters
        ----------
        mu: Tensor
            Latent space mean of encoder distribution.
        logvar: Tensor
            Latent space log variance of encoder distribution.
        """
        kl_error = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_error

    def recoLossGaussian(self, predicted_s, x, gaussian_noise_std, data_std):
        """
        Compute reconstruction loss for a Gaussian noise model.  
        This is essentially the MSE loss with a factor depending on the standard deviation.
        Parameters
        ----------
        predicted_s: Tensor
            Predicted signal by DivNoising decoder.
        x: Tensor
            Noisy observation image.
        gaussian_noise_std: float
            Standard deviation of Gaussian noise.
        data_std: float
            Standard deviation of training and validation data combined (used for normailzation).
        """
        reconstruction_error =  torch.mean((predicted_s-x)**2) / (2.0* (gaussian_noise_std/data_std)**2 )
        return reconstruction_error

    def recoLoss(self, predicted_s, x, data_mean, data_std, noiseModel):
        """Compute reconstruction loss for an arbitrary noise model. 
        Parameters
        ----------
        predicted_s: Tensor
            Predicted signal by DivNoising decoder.
        x: Tensor
            Noisy observation image.
        data_mean: float
            Mean of training and validation data combined (used for normailzation).
        data_std: float
            Standard deviation of training and validation data combined (used for normailzation).
        device: GPU device
            torch cuda device
        """
        predicted_s_denormalized = predicted_s * data_std + data_mean
        x_denormalized = x * data_std + data_mean
        predicted_s_cloned = predicted_s_denormalized
        predicted_s_reduced = predicted_s_cloned.permute(1,0,2,3)
        
        x_cloned = x_denormalized
        x_cloned = x_cloned.permute(1,0,2,3)
        x_reduced = x_cloned[0,...]
        
        likelihoods=noiseModel.likelihood(x_reduced,predicted_s_reduced)
        log_likelihoods=torch.log(likelihoods)

        # Sum over pixels and batch
        reconstruction_error= -torch.mean( log_likelihoods ) 
        return reconstruction_error
        
    def __call__(self, predicted_s, x, mu, logvar, particle_num, data_mean, data_std, noiseModel):
        """Compute DivNoising loss. 
        Parameters
        ----------
        predicted_s: Tensor
            Predicted signal by DivNoising decoder.
        x: Tensor
            Noisy observation image.
        mu: Tensor
            Latent space mean of encoder distribution.
        logvar: Tensor
            Latent space logvar of encoder distribution.
        gaussian_noise_std: float
            Standard deviation of Gaussian noise (required when using Gaussian reconstruction loss).
        data_mean: float
            Mean of training and validation data combined (used for normailzation).
        data_std: float
            Standard deviation of training and validation data combined (used for normailzation).
        device: GPU device
            torch cuda device
        noiseModel: NoiseModel object
            Distribution of noisy pixel values corresponding to clean signal (required when using general reconstruction loss).
        """
        kl_loss = self.lossFunctionKLD(mu, logvar)
        
        if noiseModel is not None:
            reconstruction_loss = self.recoLoss(predicted_s, x, data_mean, data_std, noiseModel)
        else:
            l2_loss = nn.MSELoss(reduction='sum')
            reconstruction_loss = l2_loss(predicted_s, x) / particle_num
            
            # reconstruction_loss = self.recoLossGaussian(predicted_s, x, gaussian_noise_std, data_std)
        #print(float(x.numel()))
        return reconstruction_loss, kl_loss /particle_num
