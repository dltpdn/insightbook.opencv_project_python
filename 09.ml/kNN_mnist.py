import numpy as np, cv2
import mnist

# 훈련 데이타와 테스트 데이타 가져오기 ---①
train, train_labels = mnist.getTrain()
test, test_labels = mnist.getTest()
# kNN 객체 생성 및 훈련 ---②
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# k값을 1~10까지 변경하면서 예측 ---③
for k in range(1, 11):
    # 결과 예측 ---④
    ret, result, neighbors, distance = knn.findNearest(test, k=k)
    # 정확도 계산 및 출력 ---⑤
    correct = np.sum(result == test_labels)
    accuracy = correct / result.size * 100.0
    print("K:%d, Accuracy :%.2f%%(%d/%d)" % (k, accuracy, correct, result.size) )
