import cv2
import numpy as np

img = cv2.imread('../img/morphological.png')

# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 열림 연산 적용 ---②
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)

# 결과 출력
merged = np.hstack((img, gradient))
cv2.imshow('gradient', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()