import numpy as np
import cv2

K = 16 # 군집화 갯수(16컬러) ---①
img = cv2.imread('../img/taekwonv1.jpg')
# 군집화를 위한 데이타 구조와 형식 변환 ---②
data = img.reshape((-1,3)).astype(np.float32)
# 반복 중지 요건 ---③
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 평균 클러스터링 적용 ---④
ret,label,center=cv2.kmeans(data,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# 중심 값을 정수형으로 변환 ---⑤
center = np.uint8(center)
print(center)
# 각 레이블에 해당하는 중심값으로 픽셀 값 선택 ---⑥
res = center[label.flatten()]
# 원본 영상의 형태로 변환 ---⑦
res = res.reshape((img.shape))
# 결과 출력 ---⑧
merged = np.hstack((img, res))
cv2.imshow('KMeans Color',merged)
cv2.waitKey(0)
cv2.destroyAllWindows()