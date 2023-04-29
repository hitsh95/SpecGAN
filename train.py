import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from models.models import create_model
from utils.visualizer import Visualizer
import torch
from collections import OrderedDict

opt = TrainOptions().parse()
opt.no_dropout = True
opt.no_html = True
opt.new_lr = True

val_opt = TestOptions().parse()
val_opt.phase = 'val'
val_data_loader = CreateDataLoader(val_opt)
val_dataset = val_data_loader.load_data()
val_data_size = len(val_data_loader)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0
val_total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize 
        epoch_iter = total_steps - dataset_size * (epoch - 1)

        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq== 0 and opt.model!="resnet":
            visualizer.display_current_results(model.get_current_visuals(), epoch, 10)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt.display_id, errors)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # validation
    val_freq = 1
    if epoch % val_freq == 0:
        with torch.no_grad():
            acc, val_loss = 0.0, 0.0
            print("================validation loss begain====================")
            for i,data in enumerate(val_dataset):
                iter_start_time = time.time()
                val_total_steps += opt.batchSize
                epoch_iter = val_total_steps - val_data_size * ((epoch-1)//val_freq)

                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference        
                
                if val_total_steps % opt.batchSize*20 == 0 :
                    errors = model.get_current_errors(epoch)
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                    if opt.display_id > 0  and opt.model!="resnet":
                        visualizer.plot_val_errors(epoch//val_freq, 
                                                float(epoch_iter)/val_data_size, opt.display_id + 1, errors)    
                        visualizer.display_current_results(model.get_current_visuals(), epoch, 11)

            print("================validation loss end====================")

            # resnet
            #     acc += outputs['acc_n']
            #     val_loss += outputs['val_loss']
            # val_acc = acc / val_num
            # val_loss = val_loss / val_num * opt.batchSize
            
            # val_vis = OrderedDict([('val_acc', val_acc), ('val_loss', val_loss)])

            # if opt.display_id > 0:
            #     visualizer.plot_val_errors(epoch, opt.display_id + 1, val_vis)

    if opt.new_lr:
        if epoch == opt.niter:
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):
            model.update_learning_rate()
        elif epoch == (opt.niter + 70):
            model.update_learning_rate()
        elif epoch == (opt.niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > opt.niter:
            model.update_learning_rate()




