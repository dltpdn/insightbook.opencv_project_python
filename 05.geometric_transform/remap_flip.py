import cv2
import numpy as np
import time

img = cv2.imread('../img/girl.jpg')
rows, cols = img.shape[:2]

# 뒤집기 변환 행렬로 구현 ---①
st = time.time()
mflip = np.float32([ [-1, 0, cols-1],[0, -1, rows-1]]) # 변환 행렬 생성
fliped1 = cv2.warpAffine(img, mflip, (cols, rows))     # 변환 적용
print('matrix:', time.time()-st)

# remap 함수로 뒤집기 구현 ---②
st2 = time.time()
mapy, mapx = np.indices((rows, cols),dtype=np.float32) # 매핑 배열 초기화 생성
mapx = cols - mapx -1                                  # x축 좌표 뒤집기 연산
mapy = rows - mapy -1                                  # y축 좌표 뒤집기 연산
fliped2 = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)  # remap 적용
print('remap:', time.time()-st2)

# 결과 출력 ---③
cv2.imshow('origin', img)
cv2.imshow('fliped1',fliped1)
cv2.imshow('fliped2',fliped2)
cv2.waitKey()
cv2.destroyAllWindows()