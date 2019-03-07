import numpy as np, cv2
import mnist

# 훈련 데이타 가져오기 ---①
train, train_labels = mnist.getData()
# Knn 객체 생성 및 학습 ---②
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# 인식시킬 손글씨 이미지 읽기 ---③
image = cv2.imread('../img/4027.png')
cv2.imshow("image", image)
cv2.waitKey(0) 

# 그레이 스케일 변환과 스레시홀드 ---④
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# 최외곽 컨투어만 찾기 ---⑤
img, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, \
                                        cv2.CHAIN_APPROX_SIMPLE)
# 모든 컨투어 순회 ---⑥
for c in contours:
    # 컨투어를 감싸는 외접 사각형으로 숫자 영역 좌표 구하기 ---⑦
    (x, y, w, h) = cv2.boundingRect(c)    
    # 외접 사각형의 크기가 너무 작은것은 제외 ---⑧
    if w >= 5 and h >= 25:
        # 숫자 영역만 roi로 확보하고 사각형 그리기 ---⑨
        roi = gray[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # 테스트 데이타 형식으로 변환 ---⑩
        data = mnist.digit2data(roi)
        # 결과 예측해서 이미지에 표시---⑪
        ret, result, neighbours, dist = knn.findNearest(data, k=1)
        cv2.putText(image, "%d"%ret, (x , y + 155), \
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0) 
cv2.destroyAllWindows()