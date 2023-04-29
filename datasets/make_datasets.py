import os
import numpy as np
from PIL import Image
import random
import sys
import glob
from shutil import copyfile


def make_datasets(root, folders, dirA, dirB, trainA, trainB, testA, testB):
    for folder in folders:
        data_root = os.path.join(root,folder,dirA)

        trainA = os.path.join(root, trainA)
        trainB = os.path.join(root, trainB)
        testA = os.path.join(root,testA)
        testB = os.path.join(root,testB)

        for set in os.listdir(data_root):
            n_set = set.split('/')[-1]
            raw_path = sorted(glob.glob(os.path.join(data_root, set+'/', '*.png')))
            # avg_path = [img.replace(dirA, dirB) for img in raw_path]  # paired dataset            
            for file in raw_path:
                n_file = file.split('/')[-1]
                target_name_train = trainA + f'{n_set}_{n_file}'
                target_name_test = testA + f'{n_set}_{n_file}'

                copyfile(file,target_name_train) if random.random() < 0.8 else copyfile(file,target_name_test)       
     
            # for file in avg_path:
            #     n_file = file.split('/')[-1]
            #     target_name_train = trainB + f'{n_set}_{n_file}'
            #     target_name_test = testB + f'{n_set}_{n_file}'
            #     copyfile(file,target_name_train) if random.random() < 0.8 else copyfile(file,target_name_test)     



def avg_images(root, folders, groups, S, n_images):
    idxlist = [ _ for _ in range(0, n_images)]
    for folder in folders:
        raw_path = root + folder + '/raw'
        for i in range(1, groups+1):
            raw_images = sorted(glob.glob(os.path.join(raw_path+'/'+str(i)+'/', '*' + '.png')))
            for j in range(n_images):
                ids = idxlist[:j] + idxlist[j+1:]
                ids = random.sample(ids, S-1)
                img = np.array(Image.open(raw_images[j])).astype(float)
                for idx in ids:
                    try:
                        img += np.array(Image.open(raw_images[idx])).astype(float)
                    except:
                        print(f'skip image: {raw_images[idx]}')
                img = img / S
                img = Image.fromarray(img.astype(np.uint8))

                save_dir = f'{root}{folder}/avg{S}/{i}/'
                img_name = raw_images[j].split('/')[-1]
                
                img_path = os.path.join(save_dir, img_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                img.save(img_path)



if __name__=="__main__":
    root = '/data/shahao/spms/ParticleDataset/'
    folders=['0307/']

    trainA = 'trainA/'
    trainB = 'trainB/'
    testA = 'testA/'
    testB = 'testB/'
    dirA = 'raw/'
    dirB = 'avg50/'

    # avg_images(root, folders, groups=22, S=50, n_images=60)
    make_datasets(root, folders, dirA, dirB, trainA, trainB, testA, testB)















