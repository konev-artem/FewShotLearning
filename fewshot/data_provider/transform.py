import cv2
import numpy as np
import numbers
import random


class Augmentation:
    def __init__(self, mode='train', make_flip=None, make_crop=None,
                 left_x=None, bottom_y=None, center=None, crop_size=None,
                 min_crop_size=10, make_color_jitter=None, hue=0,
                 saturation=0, value=0, alpha=1, basic_probs=None,
                 mixed_probs=None):
        self.mode = mode
        self.make_flip = make_flip
        self.make_crop = make_crop
        self.left_x = left_x
        self.bottom_y = bottom_y
        self.center = center
        self.crop_size = crop_size
        self.min_crop_size = min_crop_size
        self.make_color_jitter = make_color_jitter
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.alpha = alpha
        self.basic_transforms = [
                                 self.random_flip, self.random_crop,
                                 self.random_color_jitter
                                ]
        self.mixed_transforms = [
                                 self.mixup, self.noisy_mixup,
                                 self.between_class, self.vertical_concat,
                                 self.horizontal_concat,
                                 self.random_mixed_concat
                                ]
        def_basic_probs = np.repeat(1 / len(self.basic_transforms),
                                    len(self.basic_transforms))
        self.basic_probs = basic_probs if basic_probs else def_basic_probs
        def_mixed_probs = np.repeat(1 / len(self.mixed_transforms),
                                    len(self.mixed_transforms))
        self.mixed_probs = mixed_probs if mixed_probs else def_mixed_probs

        if isinstance(self.min_crop_size, numbers.Number):
            self.min_crop_size = (int(self.min_crop_size),
                                  int(self.min_crop_size))

    def apply_basic_transform(self, img):
        if self.make_flip:
            img = self.flip(img)
        if self.make_crop:
            if self.left_x and self.bottom_y and self.crop_size:
                if isinstance(self.crop_size, numbers.Number):
                    self.crop_size = (int(self.crop_size), int(self.crop_size))
                img = self.crop(img, self.left_x, self.bottom_y,
                                self.crop_size[0], self.crop_size[1])
            else:
                img = self.random_crop([img])
        if self.make_color_jitter:
            if self.hue or self.saturation or self.value:
                img = self.color_jitter(img, self.hue, self.saturation,
                                        self.value)
            else:
                img = self.random_color_jitter(img)
        return img

    def apply_random_basic_transform(self, images):
        for index, (transform, p) in enumerate(zip(self.basic_transforms,
                                                   self.basic_probs)):
            images = transform(images, p)
        return images

    def apply_random_mixed_transform(self, images, labels):
        f = np.random.choice(self.mixed_transforms, p=self.mixed_probs)
        return f(images, labels)

    def apply_transform(self, images, labels):
        # basic stage
        images = self.apply_basic_transform(images)
        # mixup stage
        images, labels = self.apply_random_mixed_transform(images, labels)
        images = [img.astype(dtype=np.uint8) for img in images]
        return images, labels

    def crop(self, img, left, lower, h, w):
        return img[lower:lower+h, left:left+w]

    def random_crop(self, images, p=0.5):
        if random.random() < p:
            return images
        if self.mode == 'test':
            self.center = True
        if isinstance(self.crop_size, numbers.Number):
            self.crop_size = (int(self.crop_size), int(self.crop_size))
        if self.crop_size is None:
            crop_width = np.random.randint(self.min_crop_size[1],
                                           images[0].shape[1])
            crop_height = np.random.randint(self.min_crop_size[0],
                                            images[0].shape[0])
            self.crop_size = (crop_height, crop_width)
        if self.center:
            size = (crop_height, crop_height)
            left = np.repeat(int((images[0].shape[1] - size[1]) / 2),
                             len(images))
            lower = np.repeat(int((images[0].shape[0] - size[0]) / 2),
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

    def flip(self, img):
        return cv2.flip(img, 0)

    def random_flip(self, images, p=0.5):
        images = [img if random.random() < p else self.flip(img)
                  for img in images]
        return images

    def color_jitter(self, img, hue=0, saturation=0, value=0):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(dtype=np.uint32)
        hsv[:, :, 0] += hue
        hsv[:, :, 0] %= 180
        hsv[:, :, 1] += saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] += value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def random_color_jitter(self, images, p=0.5):
        if random.random() < p:
            return images
        for index in range(len(images)):
            hue = np.random.choice(180)
            self.hue = hue
            saturation = np.random.choice(256)
            self.saturation = saturation
            value = np.random.choice(256)
            self.value = value
            images[index] = self.color_jitter(images[index], hue, saturation,
                                              value)
        return images

    '''Mixup augmentation method.
    # Reference
    - [mixup: Beyond Empirical Risk Minimization]
      (https://arxiv.org/pdf/1710.09412.pdf)
    '''
    def mixup(self, inputs, labels, alpha=1):
        coeff = np.random.beta(alpha, alpha)
        output = coeff * inputs[0] + (1 - coeff) * inputs[1]
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        return (output, y)

    def noisy_mixup(self, inputs, labels, alpha=1, scale=0.025):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        coeff += np.random.normal(scale=scale, size=(inputs[0].shape[0],
                                  inputs[0].shape[1]))
        coeff = np.clip(coeff, 0, 1)
        coeff = coeff[:, :, None]
        img = coeff * inputs[0] + (1 - coeff) * inputs[1]
        return (img, y)

    '''Between class+ augmentation method.
    # Reference
    - [Learning from Between-class Examples for Deep Sound Recognition]
      (https://arxiv.org/pdf/1711.10282.pdf)
    '''
    def between_class(self, inputs, labels, coeff=np.random.uniform(0, 1)):
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        sigma1 = inputs[0].std()
        sigma2 = inputs[1].std()
        p = 1 / (1 + sigma1 / sigma2 * (1 - coeff) / coeff)
        img = p * (inputs[0] - inputs[0].mean()) \
            + (1 - p) * (inputs[1] - inputs[1].mean())
        img /= (p**2 + (1 - p)**2)
        return (img, y)

    '''Vertical, horizontal, mixed and random mixed concats.
    # Reference
    - [Improved Mixed-Example Data Augmentation]
      (https://arxiv.org/pdf/1805.11272.pdf)
    '''
    def vertical_concat(self, inputs, labels, alpha=1):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        upper = inputs[0][:int(coeff * inputs[0].shape[0])]
        lower = inputs[1][int(coeff * inputs[1].shape[0]):]
        return (np.concatenate((upper, lower)), y)

    def horizontal_concat(self, inputs, labels, alpha=1):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * labels[0] + (1 - coeff) * labels[1]
        left = inputs[0][:, :int(coeff * inputs[0].shape[1])]
        right = inputs[1][:, int(coeff * inputs[1].shape[1]):]
        return (np.concatenate((left, right), axis=1), y)

    def mixed_concat(self, inputs, labels, order=None, alpha=1):
        if order is None:
            order = [inputs[0], inputs[1], inputs[1], inputs[0]]
        coeff1, coeff2 = np.random.beta(alpha, alpha, 2)
        height, width, channels = inputs[0].shape
        y = (coeff1 * coeff2 + (1 - coeff1) * (1 - coeff2)) * labels[0] \
            + (coeff1 * (1 - coeff2) + (1 - coeff1) * coeff2) * labels[1]
        upper_left = order[0][:int(coeff1 * height), :int(coeff2 * width)]
        upper_right = order[1][:int(coeff1 * height), int(coeff2 * width):]
        lower_left = order[2][int(coeff1 * height):, :int(coeff2 * width)]
        lower_right = order[3][int(coeff1 * height):, int(coeff2 * width):]
        upper = np.concatenate((upper_left, upper_right), axis=1)
        lower = np.concatenate((lower_left, lower_right), axis=1)
        return (np.concatenate((upper, lower)), y)

    def random_mixed_concat(self, inputs, labels, alpha=1):
        order = np.random.choice(len(inputs), 4)
        order = [inputs[i] for i in order]
        return self.mixed_concat(inputs, labels, order, alpha)
