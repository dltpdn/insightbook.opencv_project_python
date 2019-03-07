import cv2
import numpy as np
 
img = cv2.imread("../img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Good feature to trac 검출기 생성 ---①
gftt = cv2.GFTTDetector_create() 
# 키 포인트 검출 ---②
keypoints = gftt.detect(gray, None)
# 키 포인트 그리기 ---③
img_draw = cv2.drawKeypoints(img, keypoints, None)

# 결과 출력 ---④
cv2.imshow('GFTTDectector', img_draw)
cv2.waitKey(0)
cv2.destrolyAllWindows()