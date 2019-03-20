import cv2
import tensorflow as tf

class Augmentation:
    def crop(self, img, value, size, seed=None, name=None):
        return tf.image.random_crop(img, value, size, seed, name)

    def flip(self, img, direction='horizontal'):
        if direction == 'horizontal':
            return tf.image.flip_left_right(img)
        if direction == 'vertical':
            return tf.image.flip_up_down(img)
        raise ValueError('direction for flip can be horizontal or vertical')

    def color_jitter(self, img, flag=cv2.COLOR_BGR2RGB):
        return cv2.cvtColor(img, flag)

    # TODO: add more image transformations
