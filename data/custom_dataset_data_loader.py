import torch.utils.data

class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.shuffle,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'spec':
        from data.spec_dataset import SpecDataset
        dataset = SpecDataset() # 训练集的数据放在csv文件中
    elif opt.dataset_mode == 'pair':
        from data.pair_dataset import PairDataset
        dataset = PairDataset() # 配对数据集，训练集AB的图像放在两个文件夹下trainA和trainB，配对图像名称对应为low和normal
    elif opt.dataset_mode == 'syn':
        from data.syn_dataset import PairDataset
        dataset = PairDataset() # 配对数据集，低照度图像A由B*缩放因子生成，A_path没有用
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset



