import os
import copy
import pickle
from itertools import chain

import pandas as pd
from sklearn.model_selection import train_test_split

from fewshot.data_provider.generator import DataFrameIterator, FewShotDataFrameIterator
from fewshot.data_provider.transform import Augmentation


class Dataset:
    def __init__(self,
                 dataset_dir,
                 csv_name='data.csv',
                 x_col='filepath',
                 y_col='class',
                 image_size=(84,84),
                 cache=None):

        self.dataset_dir = dataset_dir
        self.csv_name = csv_name

        self.x_col = x_col
        self.y_col = y_col
        self.image_size = image_size

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
        self.class_index = dict(self.dataframe.groupby([self.y_col]).indices)

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
        dataset.dataframe = dataset.dataframe.iloc[index]
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
        train_index = list(chain(*[self.class_index[class_name] for class_name in train_classes]))
        test_index = list(chain(*[self.class_index[class_name] for class_name in test_classes]))
        return self._create_split(train_index, test_index)

    def split_by_objects(self, train_size=0.5, random_state=42):
        print('Split by objects with train size = {} (seed = {})'.format(train_size, random_state))
        train_index, test_index = zip(*([
            train_test_split(class_index, train_size=train_size, random_state=random_state) \
            for class_index in self.class_index.values()
        ]))
        train_index = list(chain(*train_index))
        test_index = list(chain(*test_index))
        return self._create_split(train_index, test_index)

    def get_batch_generator(self,
                            batch_size,
                            generator_args={},
                            **kwargs):
        return DataFrameIterator(self.dataframe,
                                 self.dataset_dir,
                                 Augmentation(**generator_args),
                                 x_col=self.x_col,
                                 y_col=self.y_col,
                                 target_size=self.image_size,
                                 batch_size=batch_size,
                                 seed=42,
                                 interpolation='bilinear',
                                 cache=self.cache,
                                 **kwargs)

    def get_fewshot_generator(self,
                              n_way,
                              k_shot,
                              query_size=None,
                              support_generator_args={},
                              query_generator_args={},
                              **kwargs):
        return FewShotDataFrameIterator(self.dataframe,
                                        self.dataset_dir,
                                        Augmentation(**support_generator_args),
                                        Augmentation(**query_generator_args),
                                        class_index=self.class_index,
                                        n_way=n_way,
                                        k_shot=k_shot,
                                        query_size=query_size,
                                        x_col=self.x_col,
                                        y_col=self.y_col,
                                        target_size=self.image_size,
                                        seed=42,
                                        interpolation='bilinear',
                                        cache=self.cache,
                                        **kwargs)
