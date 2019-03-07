import cv2
import numpy as np

# 카메라 장치 연결
cap = cv2.VideoCapture(0)   
while cap.isOpened():
    # 프레임 읽기
    ret, frame = cap.read()
    # 속도 향상을 위해 영상크기를 절반으로 축소
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, \
                        interpolation=cv2.INTER_AREA)
    if cv2.waitKey(1) == 27: # esc키로 종료
        break
    # 그레이 스케일로 변경    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 잡음 제거를 위해 가우시안 플러 필터 적용(라플라시안 필터 적용 전에 필수)
    img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
    # 라플라시안 필터로 엣지 거출
    edges = cv2.Laplacian(img_gray, -1, None, 5)
    # 스레시홀드로 경계 값 만 남기고 제거하면서 화면 반전(흰 바탕 검은 선)
    ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    
    # 경계선 강조를 위해 팽창 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    sketch = cv2.erode(sketch, kernel)
    # 경계선 자연스럽게 하기 위해 미디언 블러 필터 적용
    sketch = cv2.medianBlur(sketch, 5)
    # 그레이 스케일에서 BGR 컬러 스케일로 변경
    img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # 컬러 이미지 선명선을 없애기 위해 평균 블러 필터 적용
    img_paint = cv2.blur(frame, (10,10) )
    # 컬러 영상과 스케치 영상과 합성
    img_paint = cv2.bitwise_and(img_paint, img_paint, mask=sketch)
    
    # 결과 출력
    merged = np.hstack((img_sketch, img_paint))
    cv2.imshow('Sketch Camera', merged)
    
cap.release()
cv2.destroyAllWindows()
