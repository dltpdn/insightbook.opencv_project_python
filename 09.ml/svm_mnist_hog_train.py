import cv2 
import numpy as np
import mnist
import time

# 기울어진 숫자를 바로 세우기 위한 함수 ---①
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(20, 20),flags=affine_flags)
    return img

# HOGDescriptor를 위한 파라미터 설정 및 생성---②
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (5,5)
nbins = 9
hogDesc = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

if __name__ =='__main__':
    # MNIST 이미지에서 학습용 이미지와 테스트용 이미지 가져오기 ---③
    train_data, train_label  = mnist.getTrain(reshape=False)
    test_data, test_label = mnist.getTest(reshape=False)
    # 학습 이미지 글씨 바로 세우기 ---④
    deskewed = [list(map(deskew,row)) for row in train_data]
    # 학습 이미지 HOG 계산 ---⑤
    hogdata = [list(map(hogDesc.compute,row)) for row in deskewed]
    train_data = np.float32(hogdata)
    print('SVM training started...train data:', train_data.shape)
    # 학습용 HOG 데이타 재배열  ---⑥
    train_data = train_data.reshape(-1,train_data.shape[2])
    # SVM 알고리즘 객체 생성 및 훈련 ---⑦
    svm = cv2.ml.SVM_create()
    startT = time.time()
    svm.trainAuto(train_data, cv2.ml.ROW_SAMPLE, train_label)
    endT = time.time() - startT
    print('SVM training complete. %.2f Min'%(endT/60))  
    # 훈련된  결과 모델 저장 ---⑧
    svm.save('svm_mnist.xml')

    # 테스트 이미지 글씨 바로 세우기 및 HOG 계산---⑨
    deskewed = [list(map(deskew,row)) for row in test_data]
    hogdata = [list(map(hogDesc.compute,row)) for row in deskewed]
    test_data = np.float32(hogdata)
    # 테스트용 HOG 데이타 재배열 ---⑩
    test_data = test_data.reshape(-1,test_data.shape[2])
    # 테스트 데이타 결과 예측 ---⑪
    ret, result = svm.predict(test_data)
    # 예측 결과와 테스트 레이블이 맞은 갯수 합산 및 정확도 출력---⑫
    correct = (result==test_label).sum()
    print('Accuracy: %.2f%%'%(correct*100.0/result.size))