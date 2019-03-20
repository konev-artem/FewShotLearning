import cv2
import numpy as np

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

    def mixup(self, img1, img2, y1, y2, alpha=1):
        coeff = np.random.beta(alpha, alpha)
        img = coeff * img1 + (1 - coeff) * img2
        y = coeff * y1 + (1 - coeff) * y2
        return (img, y)
    
    def vertical_concat(self, img1, img2):
        pass
    
    def horizontal_concat(self, img1, img2):
        pass