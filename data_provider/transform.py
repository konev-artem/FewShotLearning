import cv2
import tensorflow as tf

class Augmentation:
    def crop(self, img, left, lower, h, w):
        return img[lower:lower+h, left:left+w]

    def flip(self, img, direction='horizontal'):
        if direction == 'horizontal':
            return cv2.flip(img, 0)
        if direction == 'vertical':
            return cv2.flip(img, 1)
        raise ValueError('direction for flip can be horizontal or vertical')

    def color_jitter(self, img, flag=cv2.COLOR_BGR2RGB):
        return cv2.cvtColor(img, flag)

    # TODO: add more image transformations
