
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'gan':
        # assert(opt.dataset_mode == 'unaligned')
        from .gan_model import GANModel
        model = GANModel() # EnlightenGAN,加了gan,path_d,fcn loss
    elif opt.model == 'vae':
        from .vae_model import VAEModel
        model = VAEModel()
    elif opt.model == 'resnet':
        from .resnet_model import ResnetModel
        model = ResnetModel()
    elif opt.model == 'sgan':
        from .sgan_model import sGANModel
        model = sGANModel()
    
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
