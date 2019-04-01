import os
from os.path import join, exists, splitext

import time
from shutil import move, rmtree

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import cv2
from PIL import Image

import struct
import zipfile
import tarfile
import gzip


class DatasetPreparer:
    def __init__(self, dataset_name, dataset_root):
        print('Preparing {}...'.format(dataset_name.upper()))
        self.dataset_dir = join(dataset_root, dataset_name)
          
    def download_dataset(self):
        pass

    def uncompress(self):
        pass

    def create_dataframe(self):
        pass

    def save_dataframe(self, data):
        dataframe = pd.DataFrame(data, columns=['filepath', 'class'])
        path_to_save = join(self.dataset_dir, 'data.csv')
        dataframe.to_csv(path_to_save, index=False)
        print('Save dataframe to {}'.format(path_to_save))
        
    def prepare_dataset(self):
        print('Step 1 / 3: downloading dataset (this may take a while)...')
        print('WARNING: all existing folders will be overwritten')
        for i in reversed(range(11)):
            print('New download starts in {:02d} s (press "^C" to exit)'.format(i), end='\r')
            time.sleep(1)
        print()
        self.download_dataset()
        
        print('Step 2 / 3: uncompressing dataset (this may take a while)...')
        self.uncompress()
        
        print('Step 3 / 3: creating dataframe...')
        self.create_dataframe()
        print('Completed')

        
class OmniglotPreparer(DatasetPreparer):
    
    def download_dataset(self):
        repo = 'https://github.com/brendenlake/omniglot'
        local_repo = '/tmp/omniglot'
        download_cmd = 'git clone {repo} {local_repo}'.format(repo=repo, local_repo=local_repo)
        os.system(download_cmd)

        os.makedirs(self.dataset_dir, exist_ok=True)
        for subset in ('images_background', 'images_evaluation'):
            src = join(local_repo, 'python', subset + '.zip')
            dst = join(self.dataset_dir, subset) + '.zip'
            move(src, dst)

        rmtree(local_repo)

    def uncompress(self):
        for subset in ('images_background', 'images_evaluation'):
            dataset_subdir = join(self.dataset_dir, subset)
            zip_ref = zipfile.ZipFile(dataset_subdir + '.zip', 'r')
            zip_ref.extractall(self.dataset_dir)
            zip_ref.close()
            os.remove(dataset_subdir + '.zip')             
            
    def create_dataframe(self):
        data = []
        for subset in ('images_background', 'images_evaluation'):
            dataset_subdir = join(self.dataset_dir, subset)
            alphabet_names = os.listdir(dataset_subdir)
            for alphabet_name in alphabet_names:
                alphabet_dir = join(dataset_subdir, alphabet_name)
                for char_name in os.listdir(alphabet_dir):
                    alphabet_char_dir = join(alphabet_dir, char_name)
                    for filename in os.listdir(alphabet_char_dir):
                        filepath = os.path.join(subset, alphabet_name, char_name, filename)
                        class_name = alphabet_name + '__' + char_name
                        data.append([filepath, class_name, alphabet_name, subset[len('images_'):]])
                        
        self.save_dataframe(data)

    def save_dataframe(self, data):
        dataframe = pd.DataFrame(data, columns=['filepath', 'class', 'super_class', 'subset'])
        path_to_save = join(self.dataset_dir, 'data.csv')
        dataframe.to_csv(path_to_save, index=False)
        print('Save dataframe to {}'.format(path_to_save))

        
class MiniImageNetPreparer(DatasetPreparer):
    
    def download_dataset(self):
        cookies_file = '/tmp/cookies.txt'
        confirm_cmd = "wget --quiet --save-cookies {cookies_file} --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p'".format(cookies_file=cookies_file)
        confirm_code = "$({confirm_cmd})".format(confirm_cmd=confirm_cmd)
        url = "https://docs.google.com/uc?export=download&confirm={confirm_code}&id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE".format(
            confirm_code=confirm_code)
        download_cmd = 'wget --load-cookies {cookies_file} "{url}" -O {dataset_dir}.zip && rm -rf {cookies_file}'.format(
            url=url, dataset_dir=self.dataset_dir, cookies_file=cookies_file)
        os.system(download_cmd)

    def uncompress(self):
        zip_ref = zipfile.ZipFile(self.dataset_dir + '.zip', 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()
        os.remove(self.dataset_dir + '.zip') 

    def create_dataframe(self):
        data = []
        filenames = sorted(filter(lambda x: x.endswith('.jpg'), os.listdir(join(self.dataset_dir, 'images'))))
        for filename in tqdm(filenames, total=len(filenames)):
            image_name, ext = splitext(filename)
            class_name = image_name[:-8]
            filepath = join('images', filename)
            data.append([filepath, class_name])
            
        self.save_dataframe(data)
        
        
class CUBPreparer(DatasetPreparer):
    
    def download_dataset(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz'
        download_cmd = 'wget "{url}" -O {dataset_dir}.tar.gz'.format(url=url, dataset_dir=self.dataset_dir)
        os.system(download_cmd)
        
    def uncompress(self):
        tar_ref = tarfile.open(self.dataset_dir + '.tar.gz', "r:gz")
        tar_ref.extractall(self.dataset_dir)
        tar_ref.close()
        os.remove(self.dataset_dir + '.tar.gz')

    def create_dataframe(self):
        data = []
        class_names = sorted(filter(lambda x: not x.startswith('.'), os.listdir(join(self.dataset_dir, 'images'))))
        for class_name in class_names:
            class_dir = join(self.dataset_dir, 'images', class_name)
            filenames = sorted(filter(lambda x: not x.startswith('.'), os.listdir(class_dir)))
            for filename in filenames:
                filepath = os.path.join('images', class_name, filename)
                data.append([filepath, class_name])
            
        self.save_dataframe(data)        

        
class EMNISTPreparer(DatasetPreparer):
    
    def download_dataset(self):
        url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
        download_cmd = 'wget "{url}" -O {dataset_dir}.zip'.format(url=url, dataset_dir=self.dataset_dir)
        os.system(download_cmd)

    def uncompress(self):
        zip_ref = zipfile.ZipFile(self.dataset_dir + '.zip', 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()
        os.remove(self.dataset_dir + '.zip')
        
        for dataset in ('balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'):
            for subset in ('train', 'test'):
                print('{}-{}'.format(dataset, subset))
                self._extract_images_and_labels(dataset, subset)
                print()
                
        rmtree(join(self.dataset_dir, 'gzip'))

    def _extract_images_and_labels(self, dataset, subset):
        subdir = join(self.dataset_dir, dataset, subset)
        os.makedirs(subdir, exist_ok=True)
        self._extract_images(join(self.dataset_dir, 'gzip', 'emnist-{}-{}-images-idx3-ubyte.gz'.format(dataset, subset)), 
                             join(subdir, 'images'))
        self._extract_labels(join(self.dataset_dir, 'gzip', 'emnist-{}-{}-labels-idx1-ubyte.gz'.format(dataset, subset)), 
                             join(subdir, 'labels.npy'))
                
    def _extract_labels(self, label_filename, path_to_save):
        print('Unpacking {}...'.format(label_filename))
        with gzip.open(label_filename, 'rb') as label_file:
            data = np.frombuffer(label_file.read(), np.uint8, offset=8)
        np.save(path_to_save, data)
        print('Unpacked {} labels'.format(len(data)))
                        
    def _extract_images(self, images_filename, images_save_folder):
        os.makedirs(images_save_folder, exist_ok=True)
        print('Unpacking {}...'.format(images_filename))
        with gzip.open(images_filename, 'rb') as images_file:
            images_file.read(16)
            count = 0
            while True:
                image = np.frombuffer(images_file.read(784), np.uint8)
                if len(image) == 0:
                    print('Unpacked {} images'.format(count))
                    break
                
                image = np.expand_dims(image.reshape((28, 28)).T, -1)
                path_to_save = join(images_save_folder, '{}.png'.format(count))
                cv2.imwrite(path_to_save, image)
                print(count, end='\r')
                count += 1         
        
    def create_dataframe(self):
        dataset_dir = self.dataset_dir
        for dataset in ('balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'):
            old_dataset_dir = join(self.dataset_dir, dataset)
            new_dataset_dir = '{}_{}'.format(self.dataset_dir, dataset)
            try:
                rmtree(new_dataset_dir)
            except:
                pass
            move(old_dataset_dir, new_dataset_dir)
            self.dataset_dir = new_dataset_dir
            
            data = []
            for subset in ('train', 'test'):
                subset_dir = join(self.dataset_dir, subset)
                filenames = os.listdir(join(subset_dir, 'images'))
                labels = np.load(join(subset_dir, 'labels.npy'))
                assert len(labels) == len(filenames)    
                for filename, label in zip(filenames, labels):
                    filepath = os.path.join(subset, 'images', filename)
                    index = int(splitext(filename)[0])
                    label = labels[index]
                    data.append([filepath, label])

            self.save_dataframe(data)
            self.dataset_dir = dataset_dir
        
        rmtree(self.dataset_dir)
       
        
def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('dataset', type=str, default='mini_imagenet', 
                        choices=['mini_imagenet', 'omniglot', 'cub', 'emnist'],
                        help='Dataset name (default: "mini_imagenet")')
    parser.add_argument('--dataset_root', type=str, default=os.getcwd(), 
                        help='Path to dataset root (default: ".")')
    args = parser.parse_args()
    return args

        
if __name__ == '__main__':
    args = parse_args()
    
    if args.dataset == 'mini_imagenet':
        Preparer = MiniImageNetPreparer
    elif args.dataset == 'omniglot':
        Preparer = OmniglotPreparer
    elif args.dataset == 'cub':
        Preparer = CUBPreparer
    elif args.dataset == 'emnist':
        Preparer = EMNISTPreparer    
    else:
        raise NotImplemented
    
    dataset_preparer = Preparer(args.dataset, args.dataset_root)
    dataset_preparer.prepare_dataset()
