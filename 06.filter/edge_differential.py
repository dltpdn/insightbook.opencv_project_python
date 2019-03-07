import cv2
import numpy as np

img = cv2.imread("../img/sudoku.jpg")

#미분 커널 생성 ---①
gx_kernel = np.array([[ -1, 1]])
gy_kernel = np.array([[ -1],[ 1]])

# 필터 적용 ---②
edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)
# 결과 출력
merged = np.hstack((img, edge_gx, edge_gy))
cv2.imshow('edge', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()