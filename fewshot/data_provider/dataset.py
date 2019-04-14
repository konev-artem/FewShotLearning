import os
import copy
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator as Transformer # will change for custom implementation later

from fewshot.data_provider.generator import DataFrameIterator, FewShotDataFrameIterator


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
        self._recompute_statistics()

    def __len__(self):
        return len(self.dataframe)

    def _recompute_statistics(self):
        self.n_samples = len(self.dataframe)
        self.classes = sorted(self.dataframe[self.y_col].unique())
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

    def _create_split(self, train_index, test_index):
        train_dataset = self._create_subset(train_index)
        test_dataset = self._create_subset(test_index)

        print('Train data: {} samples'.format(len(train_dataset)))
        print('Test data:  {} samples'.format(len(test_dataset)))

        return train_dataset, test_dataset

    def _create_subset(self, index):
        dataset = copy.deepcopy(self)
        dataset.dataframe = self.dataframe.loc[index]
        dataset._recompute_statistics()
        return dataset

    def split_by_classes(self, train_classes=None, test_classes=None, train_size=0.5, random_state=42):
        if train_classes is None or test_classes is None:
            print('Split by classes with train size = {} (seed = {})'.format(train_size, random_state))
            train_classes, test_classes = train_test_split(self.classes,
                                                           train_size=train_size,
                                                           random_state=random_state)
        print('Train classes: {}'.format(len(train_classes)))
        print('Test classes: {}'.format(len(test_classes)))

        train_index = list(self.dataframe[self.dataframe[self.y_col].isin(train_classes)].index)
        print(train_index)
        test_index = self.dataframe[self.dataframe[self.y_col].isin(test_classes)].index
        return self._create_split(train_index, test_index)

    def split_by_objects(self, train_size=0.5, random_state=42):
        print('Split by objects with train size = {} (seed = {})'.format(train_size, random_state))
        train_index = []
        test_index = []
        for class_name in self.classes:
            class_index = list(self.dataframe[self.dataframe[self.y_col]==class_name].index) #TODO: do it more efficiently
            class_train_index, class_test_index = train_test_split(class_index,
                                                                   train_size=train_size,
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
