import cv2
import numpy as np
import numbers
import random


class Augmentation:
    def __init__(self, mode='train', make_flip=None, make_crop=None,
                 left_x=None, bottom_y=None, center=None, crop_size=None,
                 make_color_jitter=None, hue=0, saturation=0, value=0,
                 alpha=1):
        self.mode = mode
        self.make_flip = make_flip
        self.make_crop = make_crop
        self.left_x = left_x
        self.bottom_y = bottom_y
        self.center = None
        self.crop_size = crop_size
        self.make_color_jitter = make_color_jitter
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.alpha = alpha

    def apply_classic_transform(self, img):
        if self.make_flip:
            img = self.flip(img)
        if self.make_crop:
            if self.left_x and self.bottom_y and self.crop_size:
                if isinstance(self.crop_size, numbers.Number):
                    self.crop_size = (int(self.crop_size), int(self.crop_size))
                img = self.crop(img, self.left_x, self.bottom_y,
                                self.crop_size[0], self.crop_size[1])
            else:
                img = self.random_crop(img, self.crop_size, self.center)
        if self.make_color_jitter:
            if self.hue or self.saturation or self.value:
                img = self.color_jitter(img, self.hue, self.saturation,
                                        self.value)
            else:
                img = self.random_color_jitter(img)
        return img

    def apply_random_classic_transform(self, img, p=[0.5, 0.5, 0.5]):
        transforms = [self.flip, self.random_crop, self.random_color_jitter]
        for index, transform in enumerate(transforms):
            if random.random() < p[index]:
                img = transform(img)
        return img

    def apply_random_mixed_transform(self, img1, img2, y1, y2, p=None):
        transforms = [
            self.mixup, self.noisy_mixup, self.between_class,
            self.vertical_concat, self.horizontal_concat,
            self.random_mixed_concat
            ]
        f = np.random.choice(transforms, p=p)
        img = f(img1, img2, y1, y2)
        return img

    def crop(self, img, left, lower, h, w):
        return img[lower:lower+h, left:left+w]

    def random_crop(self, img, size=None, center=False):
        # TODO: add constrains on crop size
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        if size is None:
            crop_width = np.random.randint(1, img.shape[1])
            crop_height = np.random.randint(1, img.shape[0])
            size = (crop_height, crop_width)
        if center:
            size = (crop_height, crop_height)
            left = int((img.shape[1] - size[1]) / 2)
            lower = int((img.shape[0] - size[0]) / 2)
        else:
            left = np.random.randint(img.shape[1] - size[1])
            lower = np.random.randint(img.shape[0] - size[0])
        self.crop_size = size
        self.left_x = left
        self.bottom_y = lower
        return self.crop(img, left, lower, size[0], size[1])

    def flip(self, img):
        return cv2.flip(img, 0)

    def color_jitter(self, img, hue=0, saturation=0, value=0):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(dtype=np.uint32)
        hsv[:, :, 0] += hue
        hsv[:, :, 0] %= 180
        hsv[:, :, 1] += saturation
        hsv[:, :, 1] %= 256
        hsv[:, :, 2] += value
        hsv[:, :, 2] %= 256
        hsv = hsv.astype(dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def random_color_jitter(self, img):
        hue = np.random.choice(180)
        self.hue = hue
        saturation = np.random.choice(256)
        self.saturation = saturation
        value = np.random.choice(256)
        self.value = value
        return self.color_jitter(img, hue, saturation, value)

    def mixup(self, input1, input2, y1, y2, alpha=1):
        """ Implemented mixup from mixup: Beyond Empirical Risk Minimization
            by H. Zhang et al"""
        coeff = np.random.beta(alpha, alpha)
        output = coeff * input1 + (1 - coeff) * input2
        y = coeff * y1 + (1 - coeff) * y2
        return (output, y)

    def _constrain(self, x):
        return max(min(x, 1), 0)

    def noisy_mixup(self, img1, img2, y1, y2, alpha=1, scale=0.025):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * y1 + (1 - coeff) * y2
        coeff += np.random.normal(scale=scale, size=(150, 150))
        coeff = np.vectorize(self._constrain)(coeff)
        coeff = coeff[:, :, None]
        img = coeff * img1 + (1 - coeff) * img2
        return (img, y)

    def between_class(self, img1, img2, y1, y2, coeff=np.random.uniform(0, 1)):
        """ Implemented between_class+ from Learning from Between-class Examples
            for Deep Sound Recognition by Tokozume et al"""
        y = coeff * y1 + (1 - coeff) * y2
        sigma1 = img1.std()
        sigma2 = img2.std()
        p = 1 / (1 + sigma1 / sigma2 * (1 - coeff) / coeff)
        img = p * (img1 - img1.mean()) + (1 - p) * (img2 - img2.mean())
        img /= (p**2 + (1 - p)**2)
        return (img, y)

    def vertical_concat(self, img1, img2, y1, y2, alpha=1):
        """ Implemented vertical concat from Improved Mixed-Example Data Augmentation
        by Cecilia Summers and Michael J. Dinneen"""
        coeff = np.random.beta(alpha, alpha)
        y = coeff * y1 + (1 - coeff) * y2
        upper = img1[:int(coeff * img1.shape[0])]
        lower = img2[int(coeff * img2.shape[0]):]
        return (np.concatenate((upper, lower)), y)

    def horizontal_concat(self, img1, img2, y1, y2, alpha=1):
        """ Implemented horizontal concat from Improved Mixed-Example Data Augmentation
        by Cecilia Summers and Michael J. Dinneen"""
        coeff = np.random.beta(alpha, alpha)
        y = coeff * y1 + (1 - coeff) * y2
        left = img1[:, :int(coeff * img1.shape[1])]
        right = img2[:, int(coeff * img2.shape[1]):]
        return (np.concatenate((left, right), axis=1), y)

    def mixed_concat(self, img1, img2, y1, y2, order=None, alpha=1):
        """ Implemented mixed concat from Improved Mixed-Example Data Augmentation
        by Cecilia Summers and Michael J. Dinneen"""
        if order is None:
            order = [img1, img2, img2, img1]
        coeff1, coeff2 = np.random.beta(alpha, alpha, 2)
        height, width, channels = img1.shape
        y = (coeff1 * coeff2 + (1 - coeff1) * (1 - coeff2)) * y1 \
            + (coeff1 * (1 - coeff2) + (1 - coeff1) * coeff2) * y2
        upper_left = order[0][:int(coeff1 * height), :int(coeff2 * width)]
        upper_right = order[1][:int(coeff1 * height), int(coeff2 * width):]
        lower_left = order[2][int(coeff1 * height):, :int(coeff2 * width)]
        lower_right = order[3][int(coeff1 * height):, int(coeff2 * width):]
        upper = np.concatenate((upper_left, upper_right), axis=1)
        lower = np.concatenate((lower_left, lower_right), axis=1)
        return (np.concatenate((upper, lower)), y)

    def random_mixed_concat(self, img1, img2, y1, y2, alpha=1):
        order = np.random.choice([img1, img2], 4)
        return self.mixed_concat(img1, img2, y1, y2, order, alpha)
