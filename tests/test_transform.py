import unittest

import numpy as np

from fewshot.data_provider.transform import Augmentation


class TestAugmentation(unittest.TestCase):

    def test_zero_probs(self):
        img = np.random.uniform(0, 256, (50, 50, 3))
        img = np.array([img], dtype='uint8')
        augm = Augmentation()
        res_image, res_label = augm.apply_random_transform(img)
        self.assertTrue(np.array_equal(img, [res_image]))

    def test_crop(self):
        img = np.random.uniform(0, 256, (50, 50, 3))
        img = np.array([img], dtype='uint8')
        augm = Augmentation(crop_size=10)
        res_image, res_label = augm.apply_random_transform(img)
        self.assertEqual(res_image.shape, (10, 10, 3))

    def test_concats(self):
        img1 = np.ones((2, 2), dtype='uint8')
        img2 = np.zeros((2, 2), dtype='uint8')
        images = np.array([img1, img2])
        labels = np.array([[0, 1], [1, 0]])
        augm = Augmentation(mixing_coeff=0.5, num_images_in_mixup=2, vertical_concat_prob=1)
        res_image, res_label = augm.apply_random_transform(images, labels)
        true_image = np.array([[1, 1], [0, 0]], dtype='uint8')
        true_label = np.array([0.5, 0.5])
        self.assertTrue(np.array_equal(true_image, res_image))
        self.assertTrue(np.array_equal(true_label, res_label))

        res_image, res_label = augm.horizontal_concat(images, labels, mixing_coeff=0.5)
        true_image = np.array([[1, 0], [1, 0]], dtype='uint8')
        self.assertTrue(np.array_equal(true_image, res_image))
        self.assertTrue(np.array_equal(true_label, res_label))

        res_image, res_label = augm.mixed_concat(images, labels, mixing_coeff=0.5)
        true_image = np.array([[1, 0], [0, 1]], dtype='uint8')
        self.assertTrue(np.array_equal(true_image, res_image))
        self.assertTrue(np.array_equal(true_label, res_label))

    def test_mixup(self):
        img1 = np.array([[250, 250], [250, 250]], dtype='uint8')
        img2 = np.zeros((2, 2), dtype='uint8')
        images = [img1, img2]
        labels = np.array([[0, 1], [1, 0]])
        augm = Augmentation(mixing_coeff=0.5, num_images_in_mixup=2, mixup_prob=1)
        res_image, res_label = augm.apply_random_transform(images, labels)
        true_image = np.array([[125, 125], [125, 125]], dtype='uint8')
        true_label = np.array([0.5, 0.5])
        self.assertTrue(np.array_equal(true_image, res_image))
        self.assertTrue(np.array_equal(true_label, res_label))
