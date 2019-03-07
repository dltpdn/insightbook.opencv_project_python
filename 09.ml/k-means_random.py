import numpy as np, cv2
import matplotlib.pyplot as plt

# 0~150 임의의 2수, 25개 ---①
a = np.random.randint(0,150,(25,2))
# 128~255 임의의 2수, 25개  ---②
b = np.random.randint(128, 255,(25,2))
# a, b를 병합 ---③
data = np.vstack((a,b)).astype(np.float32)
# 중지 요건 ---④
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 평균 클러스터링 적용 ---⑤
ret,label,center=cv2.kmeans(data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# label에 따라 결과 분류 ---⑥
red = data[label.ravel()==0]
blue = data[label.ravel()==1]

# plot에 결과 출력 ---⑦
plt.scatter(red[:,0],red[:,1], c='r')
plt.scatter(blue[:,0],blue[:,1], c='b')
# 각 그룹의 중앙점 출력 ---⑧
plt.scatter(center[0,0],center[0,1], s=100, c='r', marker='s')
plt.scatter(center[1,0],center[1,1], s=100, c='b', marker='s')
plt.show()