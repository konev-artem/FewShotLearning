import cv2
import numpy as np

class Augmentation:
    def get_random_transform(self, img):
        pass

    def apply_transform(self, img, transform='flip'):
        return {
            'flip': self.flip,
            'color_jitter': self.color_jitter,
            'crop': self.crop
        }[transform](self, img)

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

    def mixup(self, img1, img2, y1, y2, alpha=1):
        coeff = np.random.beta(alpha, alpha)
        img = coeff * img1 + (1 - coeff) * img2
        y = coeff * y1 + (1 - coeff) * y2
        return (img, y)
    
    def noisy_mixup(self, img1, img2, y1, y2, alpha=1, scale=0.025):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * y1 + (1 - coeff) * y2
        coeff += np.random.normal(scale=scale, size=(150, 150))
        constrain = lambda x : max(min(x, 1), 0)
        coeff = np.vectorize(constrain)(coeff)
        coeff = coeff[:, :, None]
        img = coeff * img1 + (1 - coeff) * img2
        return (img, y)

    def between_class(self, img1, img2, y1, y2, coeff=1):
        pass
    
    def vertical_concat(self, img1, img2, y1, y2, alpha=1):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * y1 + (1 - coeff) * y2
        upper = img1[:int(coeff * img1.shape[0])]
        lower = img2[int(coeff * img2.shape[0]):]
        return (np.concatenate((upper, lower)), y)
    
    def horizontal_concat(self, img1, img2, y1, y2, alpha=1):
        coeff = np.random.beta(alpha, alpha)
        y = coeff * y1 + (1 - coeff) * y2
        left = img1[:, :int(coeff * img1.shape[1])]
        right = img2[:, int(coeff * img2.shape[1]):]
        return (np.concatenate((left, right), axis=1), y)
    
    def mixed_concat(self, img1, img2, y1, y2, coeff=1):
        pass