import cv2
import numpy as np
import matplotlib.pylab as plt
 
# 0~158 구간 임의의 수 25 x 2 생성   ---①
a = np.random.randint(0,158,(25,2))
# 98~255 구간 임의의 수 25 x 2 생성 ---②
b = np.random.randint(98, 255,(25,2))
# a, b를 병합, 50 x 2의 임의의 수 생성 ---③
trainData = np.vstack((a, b)).astype(np.float32)
# 0으로 채워진 50개 배열 생성 ---④
responses = np.zeros((50,1), np.int32)
# 25 ~ 50 까지 1로 변경 ---⑤
responses[25:] = 1

# 0과 같은 자리의 학습 데이타는 빨강색 삼각형으로 분류 및 표시 ---⑥
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')
# 1과 같은 자리의 학습 데이타는 파랑색 사각형으로 분류 및 표시 ---⑦
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')
# 0~255 구간의 새로운 임의의 수 생성 및 초록색 원으로 표시 ---⑧
newcomer = np.random.randint(0,255,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
# SVM 알고리즘 객체 생성 및 훈련---⑨
svm = cv2.ml.SVM_create()

svm.trainAuto(trainData, cv2.ml.ROW_SAMPLE, responses)
# svm_random.xml 로 저장 ---⑩
svm.save('./svm_random.xml')
# 저장한 모델을 다시 읽기 ---⑪
svm2  = cv2.ml.SVM_load('./svm_random.xml')
# 새로운 임의의 수 예측 ---⑫
ret, results = svm2.predict(newcomer)
# 결과 표시 ---⑬
plt.annotate('red' if results[0]==0 else 'blue', xy=newcomer[0], xytext=(newcomer[0]+1))
print("return:%s, results:%s"%(ret, results))
plt.show()