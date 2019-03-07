import cv2
import numpy as np

img = cv2.imread("../img/salt_pepper_noise.jpg")

# 미디언 블러 적용 --- ①
blur = cv2.medianBlur(img, 5)

# 결과 출력 
merged = np.hstack((img,blur))
cv2.imshow('media', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()