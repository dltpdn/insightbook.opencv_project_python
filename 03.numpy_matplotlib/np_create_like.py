import cv2
import numpy as np

img = cv2.imread('./img/girl.jpg')
a = np.empty_like(img)
b = np.zeros_like(img)
c = np.ones_like(img)
d = np.full_like(img, 255)

print(a, a.shape, a.dtype)