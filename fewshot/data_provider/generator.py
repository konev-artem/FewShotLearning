import os
import psutil

import numpy as np

from tensorflow.keras.preprocessing import image as keras_img
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.python.keras.utils import to_categorical
from keras_preprocessing.image.iterator import BatchFromFilesMixin


class DataFrameIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading multiple images from a directory on disk and custom data
        through a dataframe.
    # Arguments
        dataframe: Pandas dataframe containing the relative paths of the
                   images in some columns and raw data in other columns
        directory: Path to the directory with images.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        x_col: Column or list of columns in dataframe with input data.
        y_col: Column or list of columns in dataframe with label data.
        target_size: tuple of integers, dimensions to resize input images to.
        batch_size: Integer, size of a batch.
        n_mix: Number of batches generated to produce a single transformed batch (used for mixup-type augmentations)
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: tensor dtype for images
        return_ino: Boolean, whether to return information about batch as (dataframe indice, class names, image names)
        cache: None if not using cache, path to dumped dictionary in JSON format otherwise. If path doesn't exist dictionary
            will be initialized and filled.
        max_memory_used: limit of virtual memory used to save cache, from 0.0 to 1.0
        min_space_available: minimal space available in bytes
    """
    def __init__(self,
                 dataframe,
                 directory,
                 image_data_generator,
                 x_col,
                 y_col,
                 target_size=(224, 224),
                 batch_size=32,
                 n_mix=1,
                 shuffle=False,
                 seed=42,
                 interpolation='bilinear',
                 data_format='channels_last',
                 subset=None,
                 dtype='float32',
                 return_info=False,
                 cache=None,
                 max_memory_used=0.5,
                 min_space_available=500*1024*1024):

        super(DataFrameIterator, self).set_processing_attrs(image_data_generator,
                                                            target_size,
                                                            'rgb',
                                                            data_format,
                                                            save_to_dir=False,
                                                            save_prefix='',
                                                            save_format='png',
                                                            subset=subset,
                                                            interpolation=interpolation)
        assert (set((x_col, y_col))) <= set(dataframe.columns)

        self.x = dataframe[x_col].astype(str)
        self.y = dataframe[y_col]
        self.index = dataframe.index.values

        self.classes = sorted(self.y.unique())
        self.n_classes = len(self.classes)
        self.encoding = {class_name: to_categorical(class_index, num_classes=self.n_classes) \
                         for class_index, class_name in enumerate(self.classes)}

        self.num_samples = len(dataframe)
        self.directory = directory
        self.dtype = dtype

        self.return_info = return_info

        self.use_cache = cache is not None
        self.set_cache(cache)

        self.max_memory_used = max_memory_used
        self.min_space_available = min_space_available

        self.n_mix = n_mix

        super(DataFrameIterator, self).__init__(
            self.num_samples,
            batch_size,
            shuffle,
            seed)

    def set_cache(self, cache):
        if self.use_cache:
            assert isinstance(cache, dict)
            if len(cache) == 0:
                print('Set empty cache')
        self.cache = cache

    def memory_safe(self):
        mem = psutil.virtual_memory()
        return (mem.percent < self.max_memory_used * 100. and mem.available > self.min_space_available)

    def _get_image(self, filename):
        filepath = os.path.join(self.directory, filename)
        if self.use_cache and filepath in self.cache:
            img_arr = self.cache[filepath]
        else:
            img = keras_img.load_img(filepath, target_size=self.target_size, interpolation=self.interpolation)
            img_arr = keras_img.img_to_array(img)
            if self.use_cache and self.memory_safe():
                self.cache[filepath] = img_arr

        if img_arr is None:
            return None

        return img_arr.copy()

    def _transform_samples(self, x, y, image_data_generator=None):
        if image_data_generator is not None:
            seed = np.random.randint(1000000)
            params = self.image_data_generator.get_random_transform(self.target_size, seed)
            transformed_x, transformed_y = self.image_data_generator.apply_transform(x[0], params), y[0]
            #will replace three lines above with the following line:
            #transformed_x, transformed_y = self.image_data_generator.apply_transform(x, y, params)
        else:
            assert len(x) == 1
            transformed_x = x[0]
            transformed_y = y[0]
        return transformed_x, transformed_y

    def _get_batches_of_transformed_samples(self, index_arrays, image_data_generator=None):
        batch_size = len(index_arrays[0])
        batch_x = np.zeros((batch_size, self.target_size[0], self.target_size[1], 3), dtype=self.dtype)
        batch_y = np.zeros((batch_size, self.n_classes))
        for index_in_batch, index_group in enumerate(zip(*index_arrays)):
            index_group = list(index_group)
            image_group = [self._get_image(filename) for filename in self.x.iloc[index_group]]
            label_group = self.y.iloc[index_group].map(self.encoding).values
            image_arr, label = self._transform_samples(image_group, label_group, image_data_generator)
            batch_x[index_in_batch] = image_arr
            batch_y[index_in_batch] = label

        if self.return_info:
            index_array = index_arrays[0] # identify index group by its first index
            batch_info = np.stack([self.index[index_array],
                                   self.y.iloc[index_array].values,
                                   self.x.iloc[index_array].values,
                                   ]).T
            batch = (batch_x, batch_y, batch_info)
        else:
            batch = (batch_x, batch_y)
        return batch

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_arrays = [next(self.index_generator) for _ in range(self.n_mix)]
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_arrays, self.image_data_generator)


class FewShotDataFrameIterator(DataFrameIterator):
    def __init__(self,
                 dataframe,
                 directory,
                 image_data_generator,
                 query_image_data_generator,
                 n_way,
                 k_shot,
                 query_samples_per_class=None,
                 *args, **kwargs):
        super(FewShotDataFrameIterator, self).__init__(dataframe,
                                                       directory,
                                                       image_data_generator=image_data_generator,
                                                       *args, **kwargs)
        self.query_image_data_generator = query_image_data_generator
        self.n_way = n_way
        self.support_samples_per_class = k_shot
        self.query_samples_per_class = query_samples_per_class

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            sample_classes = np.random.choice(self.classes, self.n_way)
            support_index_arrays = [[] for _ in range(self.n_mix)]
            query_index_arrays = [[] for _ in range(self.n_mix)]
            for class_name in sample_classes:
                class_index = self.y[self.y==class_name].index
                support_class_multiindex = np.random.choice(
                        class_index, size=self.support_samples_per_class * self.n_mix,
                        replace=len(class_index) < self.support_samples_per_class)
                query_class_index = [index for index in class_index if not index in support_class_multiindex]
                support_class_multiindex = support_class_multiindex.reshape((self.n_mix, -1)).tolist()
                for index_in_mix, support_class_index in enumerate(support_class_multiindex):
                    if self.query_samples_per_class is not None:
                        query_size = min(self.query_samples_per_class, len(query_class_index))
                    else:
                        query_size = len(query_class_index)

                    query_class_index = np.random.choice(query_class_index, size=query_size).tolist()
                    support_index_arrays[index_in_mix] += support_class_index
                    query_index_arrays[index_in_mix] += query_class_index

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        support = self._get_batches_of_transformed_samples(support_index_arrays, self.image_data_generator)
        query = self._get_batches_of_transformed_samples(query_index_arrays, self.query_image_data_generator)
        return support, query
