import os
from os.path import join, exists, dirname
from shutil import move, rmtree

from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd


class Resizer:
    CODE = {
         "nearest": cv2.INTER_NEAREST,
         "linear": cv2.INTER_LINEAR,
         "cubic": cv2.INTER_CUBIC,
         "lanczos4": cv2.INTER_LANCZOS4
    }
    def __init__(self, dataset_name, dataset_root, image_size, method):
        print('Resizing {} into {}'.format(dataset_name.upper(), image_size))
        self.dataset_dir = join(dataset_root, dataset_name)
        self.image_size = (image_size[1], image_size[0])
        self.suffix = '{}x{}'.format(*self.image_size)
        self.interpolation = self.CODE[method]

        self.original_images_dir = 'images'
        self.resized_images_dir = 'images_{}'.format(self.suffix)
        print('Resized images will be saved to "{}"'.format(self.resized_images_dir))
        self._create_images_dir()

        self.original_csv_name = 'data.csv'
        self.resized_csv_name = 'data_{}.csv'.format(self.suffix)
        print('Dataframe will be saved to "{}"'.format(self.resized_csv_name))
        self._load_dataframe()

    def _load_dataframe(self):    
        self.dataframe = pd.read_csv(join(self.dataset_dir, self.original_csv_name))

    def _save_dataframe(self):
        self.dataframe.to_csv(join(self.dataset_dir, self.resized_csv_name), index=False)

    def _load_image(self, filepath):
        img = cv2.imread(join(self.dataset_dir, filepath))
        return img

    def _resize_image(self, img):
        return cv2.resize(img, self.image_size, interpolation=self.interpolation)

    def _create_path_to_save(self, filepath):
        path_to_save = filepath.replace(self.original_images_dir, self.resized_images_dir)
        os.makedirs(dirname(join(self.dataset_dir, path_to_save)), exist_ok=True)
        return path_to_save

    def _create_images_dir(self):
        images_dir = join(self.dataset_dir, self.resized_images_dir) 
        if exists(images_dir):
            rmtree(images_dir)
        os.makedirs(images_dir)
    
    def _save_image(self, img, path_to_save):
        cv2.imwrite(join(self.dataset_dir, path_to_save), img)

    def resize(self):
        for index, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            img = self._load_image(row.filepath)
            img = self._resize_image(img)
            path_to_save = self._create_path_to_save(row.filepath)
            image = self._save_image(img, path_to_save=path_to_save)
            self.dataframe.ix[index, 'filepath'] = path_to_save
        self._save_dataframe()


def parse_size(size_str):
    size = [int(x) for x in size_str.split(',')]
    if len(size) == 1:
        size = [size[0], size[0]]
    return tuple(size)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Resize images in dataset')
    parser.add_argument('dataset', type=str, default='mini_imagenet', 
                        choices=['mini_imagenet', 'omniglot', 'cub', 'emnist'],
                        help='Dataset name (default: "mini_imagenet")')

    parser.add_argument('size', type=str, default='84,84',
                        help='''Desired image size, as two integers 
                        separated by comma or single integer (default: "84,84")''')
    parser.add_argument('--method', type=str, default='linear',
                        choices=['nearest', 'linear', 'cubic', 'lanczos4'],
                        help='Interpolation method (default: "linear")')

    default_dataset_root = join(str(os.pardir), 'fewshot', 'datasets')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, 
                        help='Path to dataset root (default: "{}")'.format(default_dataset_root))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    resizer = Resizer(args.dataset, args.dataset_root,
                      image_size=parse_size(args.size),
                      method=args.method)
    resizer.resize()
