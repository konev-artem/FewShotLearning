import os
import copy
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from keras_preprocessing.image import ImageDataGenerator as Transformer # will change for custom implementation later
from FewShotLearning.fewshot.data_provider.generator import DataFrameIterator, FewShotDataFrameIterator


class Dataset:
    def __init__(self,
                 dataset_dir,
                 csv_name='data.csv',
                 x_col='filepath',
                 y_col='class',
                 cache=None):

        self.dataset_dir = dataset_dir
        self.csv_name = csv_name

        self.x_col = x_col
        self.y_col = y_col

        self.cache = cache
        if isinstance(self.cache, str):
            self.load_cache(self.cache)

        self.dataframe = pd.read_csv(os.path.join(self.dataset_dir, self.csv_name))
        self.n_samples = len(self.dataframe)
        self.classes = self.dataframe[self.y_col].unique()
        self.n_classes = len(self.classes)

    def load_cache(self, cache_file):
        try:
            with open(cache_file, 'rb') as cache_fp:
                self.cache = pickle.load(cache_fp)
        except:
            print('Failed to load cached images from "{}", initialized empty cache'.format(cache_file))
            self.cache = {}
        else:
            print('Successfully loaded cached images from "{}"'.format(cache_file))

    def dump_cache(self, cache_file):
        with open(cache_file, 'wb') as cache_fp:
            pickle.dump(self.cache, cache_fp)
        print('Saved cached images to "{}"'.format(cache_file))

    def _create_subdataset(self, dataframe):
        subdataset = copy.deepcopy(self)
        subdataset.dataframe = dataframe
        return subdataset

    def _create_split(self, train_index, test_index):
        train_dataframe = self.dataframe.iloc[train_index]
        test_dataframe = self.dataframe.iloc[test_index]
        print('Train data: {} samples'.format(len(train_dataframe)))
        print('Test data:  {} samples'.format(len(test_dataframe)))

        train_dataset = copy.deepcopy(self)
        train_dataset.dataframe = train_dataframe
        test_dataset = copy.deepcopy(self)
        test_dataset.dataframe = test_dataframe
        return train_dataset, test_dataset

    def split_by_classes(self, train_classes=None, test_classes=None, test_size=0.5, random_state=42):
        if train_classes is None or test_classes is None:
            print('Split by classes with test size = {} (seed = {})'.format(test_size, random_state))
            train_classes, test_classes = train_test_split(self.classes,
                                                           test_size=test_size,
                                                           random_state=random_state)
        print('Train classes: {}'.format(len(train_classes)))
        print('Test classes: {}'.format(len(test_classes)))

        train_index = self.dataframe[self.dataframe[self.y_col].isin(train_classes)].index
        test_index = self.dataframe[self.dataframe[self.y_col].isin(test_classes)].index
        return self._create_split(train_index, test_index)

    def split_by_objects(self, test_size=0.5, random_state=42):
        print('Split by objects with test size = {} (seed = {})'.format(test_size, random_state))
        train_index = []
        test_index = []
        for class_name in self.classes:
            class_index = list(self.dataframe[self.dataframe[self.y_col]==class_name].index)
            class_train_index, class_test_index = train_test_split(class_index,
                                                                   test_size=test_size,
                                                                   random_state=random_state)
            train_index += class_train_index
            test_index += class_test_index

        return self._create_split(train_index, test_index)

    def get_batch_generator(self,
                            batch_size,
                            n_mix=1,
                            generator_args={},
                            **kwargs):
        return DataFrameIterator(self.dataframe,
                                 self.dataset_dir,
                                 Transformer(**generator_args),
                                 x_col=self.x_col,
                                 y_col=self.y_col,
                                 n_mix=n_mix,
                                 batch_size=batch_size,
                                 seed=42,
                                 interpolation='bilinear',
                                 cache=self.cache,
                                 **kwargs)

    def get_few_shot_generator(self,
                               n_way,
                               k_shot,
                               query_samples_per_class=None,
                               support_generator_args={},
                               query_generator_args={},
                               **kwargs):
        return FewShotDataFrameIterator(self.dataframe,
                                        self.dataset_dir,
                                        Transformer(**support_generator_args),
                                        Transformer(**query_generator_args),
                                        n_way=n_way,
                                        k_shot=k_shot,
                                        query_samples_per_class=query_samples_per_class,
                                        x_col=self.x_col,
                                        y_col=self.y_col,
                                        seed=42,
                                        interpolation='bilinear',
                                        cache=self.cache,
                                        **kwargs)
