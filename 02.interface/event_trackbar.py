import cv2
import numpy as np

win_name = 'Trackbar'                                   # 창 이름

img = cv2.imread('../img/blank_500.jpg')
cv2.imshow(win_name,img)                                # 초기 이미지를 창에 표시

# 트랙바 이벤트 처리 함수 선언 ---①
def onChange(x):                                        
    print(x)                                            # 트랙바 새로운 위치 값 --- ②
    # 'R', 'G', 'B' 각 트랙바 위치 값    --- ③
    r = cv2.getTrackbarPos('R',win_name)               
    g = cv2.getTrackbarPos('G',win_name)               
    b = cv2.getTrackbarPos('B',win_name)               
    print(r, g, b)
    img[:] = [b,g,r]                                    # 기존 이미지에 새로운 픽셀 값 적용 --- ④
    cv2.imshow(win_name, img)                           # 새 이미지 창에 표시

# 트랙바 생성    --- ⑤
cv2.createTrackbar('R', win_name, 255, 255, onChange)  
cv2.createTrackbar('G', win_name, 255, 255, onChange)
cv2.createTrackbar('B', win_name, 255, 255, onChange)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()        