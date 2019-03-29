import cv2
import numpy as np
import random

def apply_random_classic_transform(img, p=0.5):
    transforms = [random_flip, random_crop, random_color_jitter]
    for f in transforms: 
        if random.random() < p: 
            img = f(img)
    return img

def apply_mixed_transform(img1, img2, y1, y2, p=0.5):
    transforms = [mixup, noisy_mixup, between_class, vertical_concat, horizontal_concat, mixed_concat]
    for f in transforms: 
        if random.random() < p: 
            img = f(img1, img2, y1, y2)
    return img

def crop(img, left, lower, h, w):
    return img[lower:lower+h, left:left+w]

def random_crop(img):
    left = np.random.choice(img.shape[1])
    lower = np.random.choice(img.shape[0])
    w = np.random.choice(img.shape[1] - left)
    h = np.random.choice(img.shape[0] - lower)
    return crop(img, left, lower, h, w)

def flip(img, direction='horizontal'):
    if direction == 'horizontal':
        return cv2.flip(img, 0)
    if direction == 'vertical':
        return cv2.flip(img, 1)
    raise ValueError('direction for flip can be horizontal or vertical')

def random_flip(img):
    direction = np.random.choice(2)
    return cv2.flip(img, direction)
    
def color_jitter(img, degree=0):
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(dtype=np.uint32)
    hsv[:, :, 0] += degree 
    hsv[:, :, 0] %= 180
    hsv = hsv.astype(dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_color_jitter(img):
    degree = np.random.choice(180)
    return color_jitter(img, degree)

def mixup(img1, img2, y1, y2, alpha=1):
    coeff = np.random.beta(alpha, alpha)
    img = coeff * img1 + (1 - coeff) * img2
    y = coeff * y1 + (1 - coeff) * y2
    return (img, y)

def noisy_mixup(img1, img2, y1, y2, alpha=1, scale=0.025):
    coeff = np.random.beta(alpha, alpha)
    y = coeff * y1 + (1 - coeff) * y2
    coeff += np.random.normal(scale=scale, size=(150, 150))
    constrain = lambda x : max(min(x, 1), 0)
    coeff = np.vectorize(constrain)(coeff)
    coeff = coeff[:, :, None]
    img = coeff * img1 + (1 - coeff) * img2
    return (img, y)

def between_class(img1, img2, y1, y2, coeff=np.random.uniform(0, 1)):
    sigma1 = img1.std()
    sigma2 = img2.std()
    p = 1 / (1 + sigma1 / sigma2 * (1 - coeff) / coeff)
    return (p * (img1 - img1.mean()) + (1 - p) * (img2 - img2.mean())) / (p**2 + (1 - p)**2)

def vertical_concat(img1, img2, y1, y2, alpha=1):
    coeff = np.random.beta(alpha, alpha)
    y = coeff * y1 + (1 - coeff) * y2
    upper = img1[:int(coeff * img1.shape[0])]
    lower = img2[int(coeff * img2.shape[0]):]
    return (np.concatenate((upper, lower)), y)

def horizontal_concat(img1, img2, y1, y2, alpha=1):
    coeff = np.random.beta(alpha, alpha)
    y = coeff * y1 + (1 - coeff) * y2
    left = img1[:, :int(coeff * img1.shape[1])]
    right = img2[:, int(coeff * img2.shape[1]):]
    return (np.concatenate((left, right), axis=1), y)

def mixed_concat(img1, img2, y1, y2, coeff=1):
    pass