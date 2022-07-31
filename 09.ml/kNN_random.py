import cv2,  numpy as np,  matplotlib.pyplot as plt

# 0~200 사이의 무작위 수 50x2개 데이타 생성 ---①
red = np.random.randint(0, 110, (25,2)).astype(np.float32)
blue = np.random.randint(90, 200, (25, 2)).astype(np.float32)
trainData = np.vstack((red, blue))

# 50x1개 레이블 생성 ---②
labels = np.zeros((50,1), dtype=np.float32) # 0:빨강색 삼각형
labels[25:] = 1           # 1:파랑색 사각형

# 레이블 값 0과 같은 자리는 red, 1과 같은 자리는 blue로 분류해서 표시
plt.scatter(red[:,0], red[:,1], 80, 'r', '^') # 빨강색 삼각형
plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')# 파랑색 사각형

# 0 ~ 200 사이의 1개의 새로운 무작위 수 생성 ---③
newcomer = np.random.randint(0,200,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o') # 초록색 원

# KNearest 알고리즘 객체 생성 ---④
knn = cv2.ml.KNearest_create()
# train, 행 단위 샘플 ---⑤
knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)
# 예측 ---⑥
#ret, results = knn.predict(newcomer)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)#K=3
# 결과 출력
print('ret:%s, result:%s, negibours:%s, distance:%s' \
        %(ret,results, neighbours, dist))
plt.annotate('red' if ret==0.0 else 'blue', xy=newcomer[0], \
             xytext=(newcomer[0]+1))
plt.show()