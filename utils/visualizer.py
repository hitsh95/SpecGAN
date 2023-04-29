import numpy as np
import os
import ntpath
import time
from . import util
from . import html

import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import itertools

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.model
        self.lamb = []
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.model, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.model, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, idx=10):
        if self.display_id > 0: # show images in the browser
            clsName = visuals['clsName'][0] # [B,]
            self.lamb = visuals['lambda'][0]

            real_A = visuals['real_A'][0][0] # [B,C,120]
            fake_B = visuals['fake_B'][0][0]
            real_B = visuals['real_B'][0][0]

            import matplotlib.pyplot as plt 
            fig, ax = plt.subplots()

            ax.plot(self.lamb, real_A, label='real_A', linewidth=1.2)
            ax.plot(self.lamb, fake_B, label='fake_B', linewidth=1.2)
            ax.plot(self.lamb, real_B, label='real_B', linewidth=1.2)

            ax.legend(loc='best')
            ax.set(
                xlabel='Lambda',
                ylabel='Intensity',
                title=f'spectral curves for {clsName}'
            )

            self.vis.matplot(plt, win=self.display_id + idx)
            plt.close()

            # for label in ['real_A', 'fake_B', 'real_B']:
            #     fig, ax = plt.subplots()
            #     if label not in visuals.keys():
            #         continue
            #     data = visuals[label]
            #     import matplotlib.pyplot as plt 
            #     colors = list(mcolors.TABLEAU_COLORS.keys()) 
            #     makers = list(mmarkers.MarkerStyle.markers.keys()) 
            #     ax.plot(self.lamb,data[0],color=mcolors.TABLEAU_COLORS[colors[0]],
            #     label=f'{label}', linewidth=1.2) # marker=makers[0]
                
            #     ax.legend(loc='best')
            #     ax.set(
            #         xlabel='Lambda',
            #         ylabel='Intensity',
            #         title=f'spectral curves for {clsName}'
            #     )
            #     self.vis.matplot(plt, win=self.display_id + idx)
            #     plt.close()
                # idx += 1

        if self.use_html: # save images to a html file
            for label, data in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(data, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, data in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, display_id, errors):
        if not hasattr(self, 'plot_data') :
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}

        self.plot_data['X'].append(epoch + counter_ratio)
        if len(self.plot_data['legend']) > 1:
            self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=display_id)
        else:
            self.plot_data['Y'].append(list(errors.values())[0])
            self.vis.line(
                X=np.array(self.plot_data['X']),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=display_id)
    
    # errors: dictionary of error labels and values
    def plot_val_errors(self, epoch, counter_ratio, display_id, errors):
        if not hasattr(self, 'val_plot_data') :
            self.val_plot_data = {'X':[],'Y':[], 'legend':[]}

        self.val_plot_data['X'].append(epoch + counter_ratio)
        self.val_plot_data['legend'] = list(errors.keys())

        if len(self.val_plot_data['legend']) > 1:
            self.val_plot_data['Y'].append([errors[k] for k in self.val_plot_data['legend']])
            self.vis.line(
                X=np.stack([np.array(self.val_plot_data['X'])]*len(self.val_plot_data['legend']),1),
                Y=np.array(self.val_plot_data['Y']),
                opts={
                    'title': self.name + ' val_loss over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=display_id)
        else:
            self.val_plot_data['Y'].append(list(errors.values())[0])
            self.vis.line(
                X=np.array(self.val_plot_data['X']),
                Y=np.array(self.val_plot_data['Y']),
                opts={
                    'title': self.name + ' val_loss over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, data in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(data, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    # 绘制混淆矩阵
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.figure()
        if normalize:
            cls_num = cm.sum(axis=1)[:, np.newaxis]
            cm = cm.astype('float') / np.maximum(cls_num, np.ones_like(cls_num))
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        fmt = '.2f' if normalize else 'd'
        cm = cm.T
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(i, j, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('./results/confusion_matrix.png', format='png',dpi=200, bbox_inches = 'tight')


