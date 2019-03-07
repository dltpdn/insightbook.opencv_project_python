import cv2
import numpy as np

img = np.zeros((120,120), dtype=np.uint8)   # 120x120 2차원 배열 생성, 검은색 흑백 이미지
img[25:35, :] = 45                          # 25~35행 모든 열에 45 할당 
img[55:65, :] = 115                         # 55~65행 모든 열에 115 할당 
img[85:95, :] = 160                         # 85~95행 모든 열에 160 할당 
img[:, 35:45] = 205                         # 모든행 35~45 열에 205 할당 
img[:, 75:85] = 255                         # 모든행 75~85 열에 255 할당 
cv2.imshow('Gray', img)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
