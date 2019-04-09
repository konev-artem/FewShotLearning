import cv2
import numpy as np
import numbers
import random


class Augmentation:
    def __init__(self, mode='train', flip_prob=0, crop_prob=0,
                 center=None, crop_size=None,
                 color_jitter_prob=0, hue_range=(0, 0.5), saturation_range=(0, 1.0),
                 value_range=(0, 1.0), mixup_prob=0, between_class_prob=0, vertical_concat_prob=0,
                 horizontal_concat_prob=0, mixed_concat_prob=0,
                 alpha=1):
        self.mode = mode
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.center = center
        self.crop_size = crop_size
        self.color_jitter_prob = color_jitter_prob
        self.hsv_ranges = (hue_range, saturation_range, value_range)
        self.mixup_prob = mixup_prob
        self.between_class_prob = between_class_prob
        self.vertical_concat_prob = vertical_concat_prob
        self.horizontal_concat_prob = horizontal_concat_prob
        self.mixed_concat_prob = mixed_concat_prob
        self.alpha = alpha
        self.basic_transforms = [
                                 self.random_flip, self.random_crop,
                                 self.random_color_jitter
                                ]
        self.mixed_transforms = [
                                 self.mixup, self.between_class,
                                 self.vertical_concat,
                                 self.horizontal_concat,
                                 self.mixed_concat
                                ]
        self.basic_probs = [self.flip_prob, self.crop_prob,
                            self.color_jitter_prob]
        self.mixed_probs = [self.mixup_prob, self.between_class_prob, self.vertical_concat_prob,
                            self.horizontal_concat_prob, self.mixed_concat_prob]
        self.mixup_stage = (sum(self.mixed_probs) > 0)
        if self.mode == 'test':
            self.center = True
        if isinstance(self.crop_size, numbers.Number):
            self.crop_size = (int(self.crop_size), int(self.crop_size))

    def apply_random_basic_transform(self, images):
        for index, (transform, p) in enumerate(zip(self.basic_transforms,
                                                   self.basic_probs)):
            images = transform(images, p)
        return images

    def apply_random_mixed_transform(self, images, labels):
        f = np.random.choice(self.mixed_transforms, p=self.mixed_probs)
        return f(images, labels)

    def apply_random_transform(self, images, labels=None):
        # basic stage
        images = self.apply_random_basic_transform(images)
        # mixup stage
        if self.mixup_stage:
            labels = np.asarray(labels, dtype=np.float32)
            images, labels = self.apply_random_mixed_transform(images, labels)
            images = [img.astype(dtype=np.uint8) for img in images]
        return images, labels

    def crop(self, img, left, lower, h, w):
        return img[lower:lower+h, left:left+w]

    def random_crop(self, images, p=0.5):
        if random.random() > p:
            return images
        if self.center:
            left = np.repeat(int((images[0].shape[1] - self.crop_size[1]) / 2),
                             len(images))
            lower = np.repeat(int((images[0].shape[0] - self.crop_size[0]) / 2),
                              len(images))
        else:
            left = np.random.randint(images[0].shape[1] - self.crop_size[1],
                                     size=len(images))
            lower = np.random.randint(images[0].shape[0] - self.crop_size[0],
                                      size=len(images))
        for index in range(len(images)):
            images[index] = self.crop(images[index], left[index],
                                      lower[index], self.crop_size[0],
                                      self.crop_size[1])
        return images

    def random_flip(self, images, p=0.5):
        images = [img if random.random() > p else cv2.flip(img, 0)
                  for img in images]
        return images

    def color_jitter(self, img, hsv_factors):
        hue, saturation, value = hsv_factors
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_img = hsv_img.astype(dtype=np.float32)
        hsv_img[:, :, 0] *= hue
        hsv_img[:, :, 1] *= saturation
        hsv_img[:, :, 2] *= value
        hsv_img = hsv_img.astype(dtype=np.uint8)
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    def random_color_jitter(self, images, p=0.5):
        if random.random() > p:
            return images
        hue, saturation, value = self.hsv_ranges
        for index in range(len(images)):
            hue_factor = np.random.uniform(hue[0], hue[1])
            saturation_factor = np.random.uniform(saturation[0], saturation[1])
            value_factor = np.random.uniform(value[0], value[1])
            self.hsv_factors = (hue_factor, saturation_factor, value_factor)
            images[index] = self.color_jitter(images[index], self.hsv_factors)
        return images

    def mixup(self, inputs, labels, alpha=1, noisy=False, scale=0.025):
        '''Mixup augmentation method.
        # Reference
        - [mixup: Beyond Empirical Risk Minimization]
        (https://arxiv.org/pdf/1710.09412.pdf)
        '''
        coeff = np.random.beta(alpha, alpha)
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        if noisy:
            coeff += np.random.normal(scale=scale, size=(inputs[0].shape[0],
                                      inputs[0].shape[1]))
            coeff = np.clip(coeff, 0, 1)
            coeff = coeff[:, :, None]
        img = coeff * inputs[0] + (1 - coeff) * inputs[1]
        return (img, y)

    def between_class(self, inputs, labels, coeff=np.random.uniform(0, 1)):
        '''Between class+ augmentation method.
        # Reference
        - [Learning from Between-class Examples for Deep Sound Recognition]
        (https://arxiv.org/pdf/1711.10282.pdf)
        '''
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        sigma1 = inputs[0].std()
        sigma2 = inputs[1].std()
        p = 1 / (1 + sigma1 / sigma2 * (1 - coeff) / coeff)
        img = p * (inputs[0] - inputs[0].mean()) \
            + (1 - p) * (inputs[1] - inputs[1].mean())
        img /= (p**2 + (1 - p)**2)
        return (img, y)

    def vertical_concat(self, inputs, labels, alpha=1):
        '''Vertical mixed concat.
        # Reference
        - [Improved Mixed-Example Data Augmentation]
      (https://arxiv.org/pdf/1805.11272.pdf)
        '''
        coeff = np.random.beta(alpha, alpha)
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        upper = inputs[0][:int(coeff * inputs[0].shape[0])]
        lower = inputs[1][int(coeff * inputs[1].shape[0]):]
        return (np.concatenate((upper, lower)), y)

    def horizontal_concat(self, inputs, labels, alpha=1):
        '''Horizontal mixed concat.
        # Reference
        - [Improved Mixed-Example Data Augmentation]
      (https://arxiv.org/pdf/1805.11272.pdf)
        '''
        coeff = np.random.beta(alpha, alpha)
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        left = inputs[0][:, :int(coeff * inputs[0].shape[1])]
        right = inputs[1][:, int(coeff * inputs[1].shape[1]):]
        return (np.concatenate((left, right), axis=1), y)

    def mixed_concat(self, inputs, labels, order=[0, 1, 1, 0], alpha=1):
        '''Mixed concat.
        # Reference
        - [Improved Mixed-Example Data Augmentation]
      (https://arxiv.org/pdf/1805.11272.pdf)
        '''
        coeff1, coeff2 = np.random.beta(alpha, alpha, 2)
        height, width, channels = inputs[0].shape
        y = (coeff1 * coeff2 + (1 - coeff1) * (1 - coeff2)) * labels[0] \
            + (coeff1 * (1 - coeff2) + (1 - coeff1) * coeff2) * labels[1]
        img1, y1 = self.vertical_concat([inputs[0], inputs[1]], [labels[0], labels[1]])
        img2, y2 = self.vertical_concat([inputs[1], inputs[0]], [labels[1], labels[0]])
        return self.horizontal_concat([img1, img2], [y1, y2])
