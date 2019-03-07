import cv2
import numpy as np
import matplotlib.pyplot as plt

# 0~99 사이의 랜덤 값 25x2 ---①
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
# trainDatat[0]:kick, trainData[1]:kiss, kick > kiss ? 1 : 0 ---②
responses = (trainData[:, 0] >trainData[:,1]).astype(np.float32)
# 0: action : 1romantic ---③
action = trainData[responses==0]
romantic = trainData[responses==1]
# action은 파랑 삼각형, romantic은 빨강색 동그라미로 표시 ---④
plt.scatter(action[:,0],action[:,1], 80, 'b', '^', label='action')
plt.scatter(romantic[:,0],romantic[:,1], 80, 'r', 'o',label="romantic")
# 새로운 데이타 생성, 0~99 랜덤 수 1X2, 초록색 사각형으로 표시 ---⑤
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],200,'g','s', label="new")

# Knearest 알고리즘 생성 및 훈련 --- ⑥
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
# 결과 예측 ---⑦
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)#K=3
print("ret:%s, result:%s, neighbours:%s, dist:%s" \
            %(ret, results, neighbours, dist))
# 새로운 결과에 화살표로 표시 ---⑧
anno_x, anno_y = newcomer.ravel()
label = "action" if results == 0 else "romantic" 
plt.annotate(label, xy=(anno_x + 1, anno_y+1), \
            xytext=(anno_x+5, anno_y+10), arrowprops={'color':'black'})
plt.xlabel('kiss');plt.ylabel('kick')
plt.legend(loc="upper right")
plt.show()