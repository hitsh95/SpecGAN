from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from models.models import create_model
from utils.visualizer import Visualizer
import numpy as np
import pandas as pd
import os
import time
from utils.util import confusion_matrix

opt = TestOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
dataset_size = len(data_loader)

day = time.strftime('%m%d%H%M', time.localtime())
dataName = opt.dataroot.split('/')[-2]
root_path = opt.results_dir
fileName = f'{opt.model}_{opt.which_epoch}_{day}_{dataName}_outputs.csv'
save_path = os.path.join(root_path, fileName)

total_steps = 0

if opt.model == 'resnet':
    conf_matrix = np.zeros((2,2))
else:
    conf_matrix = np.zeros((opt.num_class, opt.num_class))

for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()

    total_steps += opt.batchSize
    
    spec_idx = list(data.keys()).index('lambda')
    csvtitle = list(data.keys())[:spec_idx]

    table = np.expand_dims(data['id'].cpu().numpy().T, axis=1) #id
    for key, value in data.items():
        if key == 'id':
            continue
        if key == 'lambda':
            break
        if key == 'cls' and opt.model == 'resnet':
            label = data['cls'].cpu().numpy()
            outputs = model.get_current_visuals()
            predict_cls_A = outputs['predict_cls_A']
            noise_idx = predict_cls_A == 0
            label[noise_idx] = 10
            table = np.hstack((table, np.expand_dims(label, axis=1)))
            continue
        if isinstance(value[0], str):
            value = np.expand_dims(np.array(value).T, axis=1)
        else:
            value = np.expand_dims(value.cpu().numpy().T, axis=1)
        table = np.hstack((table, value))
    
    outputs = model.get_current_visuals()
    lamb = outputs['lambda'][0]  # [B]
    if opt.model != 'resnet':
        fakeB = np.squeeze(outputs['fake_B'], axis=1) # [B,120]
    else:
        fakeB = np.squeeze(data['fit_curves'], axis=1)

    predict_cls_A = outputs['predict_cls_A']
    label = data['cls'].cpu().numpy()
    conf_matrix = confusion_matrix(predict_cls_A, label, conf_matrix)

    csvtitle = np.hstack((csvtitle, lamb))
    table = np.hstack((table, fakeB))

    df = pd.DataFrame(table)
    df.columns = csvtitle
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(save_path):
        df.to_csv(save_path, mode='a',header=True, index=False)
    else:
        df.to_csv(save_path, mode='a',header=False, index=False)

    if total_steps % opt.print_freq == 0:
        print(f'process data... {total_steps}/{dataset_size}')


if opt.model == 'resnet':
    clsName = ['noise', 'signal']
else:
    clsName = ["sc","CHCL3","dsc","acetone","EtOH","dsc311","C3H8O","ds","dc","MeOH","noise"]
Visualizer.plot_confusion_matrix(conf_matrix, classes=clsName, normalize=True,
                                 title='Normalized confusion matrix')

