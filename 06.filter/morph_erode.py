import cv2
import numpy as np

img = cv2.imread('../img/morph_dot.png')

# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 침식 연산 적용 ---②
erosion = cv2.erode(img, k)

# 결과 출력
merged = np.hstack((img, erosion))
cv2.imshow('Erode', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()