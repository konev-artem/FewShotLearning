import cv2
import numpy as np
import random


class Augmentation:
    def apply_random_classic_transform(self, img, p=[0.5, 0.5, 0.5]):
        transforms = [self.flip, self.random_crop, self.random_color_jitter]
        for index, transform in transforms:
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

    def random_crop(self, img, crop_width=None, center=False):
        # TODO: add constrains on crop size, write code for non-square images
        if crop_width is None:
            crop_width = np.random.randint(1, img.shape[1])
        if center:
            left = int((img.shape[1] - crop_width) / 2)
        else:
            left = np.random.randint(img.shape[1] - crop_width)
        return self.crop(img, left, left, crop_width, crop_width)

    def flip(self, img):
        return cv2.flip(img, 0)

    def color_jitter(self, img, degree=0):
        # TODO: add saturation and value changes
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(dtype=np.uint32)
        hsv[:, :, 0] += degree
        hsv[:, :, 0] %= 180
        hsv = hsv.astype(dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def random_color_jitter(self, img):
        degree = np.random.choice(180)
        return self.color_jitter(img, degree)

    def mixup(self, input1, input2, y1, y2, alpha=1):
        """ Implemented mixup from mixup: Beyond Empirical Risk Minimization
            by H. Zhang et al"""
        coeff = np.random.beta(alpha, alpha)
        output = coeff * input1 + (1 - coeff) * input2
        y = coeff * y1 + (1 - coeff) * y2
        return (output, y)

    def constrain(self, x):
        return max(min(x, 1), 0)

    def noisy_mixup(self, img1, img2, y1, y2, alpha=1, scale=0.025):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * y1 + (1 - coeff) * y2
        coeff += np.random.normal(scale=scale, size=(150, 150))
        coeff = np.vectorize(self.constrain)(coeff)
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
