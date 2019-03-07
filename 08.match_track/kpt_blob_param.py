import cv2
import numpy as np
 
img = cv2.imread("../img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blob 검출 필터 파라미터 생성 ---①
params = cv2.SimpleBlobDetector_Params()

# 경계값 조정 ---②
params.minThreshold = 10
params.maxThreshold = 240
params.thresholdStep = 5
# 면적 필터 켜고 최소 값 지정 ---③
params.filterByArea = True
params.minArea = 200
  
# 컬러, 볼록 비율, 원형비율 필터 옵션 끄기 ---④
params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByCircularity = False 

# 필터 파라미터로 blob 검출기 생성 ---⑤
detector = cv2.SimpleBlobDetector_create(params)
# 키 포인트 검출 ---⑥
keypoints = detector.detect(gray)
# 키 포인트 그리기 ---⑦
img_draw = cv2.drawKeypoints(img, keypoints, None, None,\
                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 결과 출력 ---⑧
cv2.imshow("Blob with Params", img_draw)
cv2.waitKey(0)