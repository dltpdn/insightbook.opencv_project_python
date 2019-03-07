import cv2
import numpy as np

img = cv2.imread('../img/taekwonv1.jpg')
rows, cols = img.shape[:2]
img_draw = img.copy()

# 마커 생성, 모든 요소는 0으로 초기화 ---①
marker = np.zeros((rows, cols), np.int32)
markerId = 1        # 마커 아이디는 1에서 시작
colors = []         # 마커 선택한 영역 색상 저장할 공간
isDragging = False  # 드래그 여부 확인 변수

# 마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param):
    global img_draw, marker, markerId, isDragging
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운, 드래그 시작 
        isDragging = True
        # 각 마커의 아이디와 현 위치의 색상 값을 쌍으로 매핑해서 저장 
        colors.append((markerId, img[y,x]))
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임
        if isDragging:                  # 드래그 진행 중
            # 마우스 좌표에 해당하는 마커의 좌표에 동일한 마커 아이디로 채워 넣기 ---②
            marker[y,x] = markerId
            # 마커 표시한 곳을 빨강색점으로 표시해서 출력
            cv2.circle(img_draw, (x,y), 3, (0,0,255), -1)
            cv2.imshow('watershed', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼 업
        if isDragging:                  
            isDragging = False          # 드래그 중지
            # 다음 마커 선택을 위해 마커 아이디 증가 ---③
            markerId +=1
    elif event == cv2.EVENT_RBUTTONDOWN: # 오른쪽 마우스 버튼 누름
            # 모아 놓은 마커를 이용해서 워터 쉐드 적용 ---④
            cv2.watershed(img, marker)
            # 마커에 -1로 표시된 경계를 초록색으로 표시  ---⑤
            img_draw[marker == -1] = (0,255,0)
            for mid, color in colors: # 선택한 마커 아이디 갯수 만큼 반복
                # 같은 마커 아이디 값을 갖는 영역을 마커 선택한 색상으로 채우기 ---⑥
                img_draw[marker==mid] = color
            cv2.imshow('watershed', img_draw) # 표시한 결과 출력

# 화면 출력
cv2.imshow('watershed', img)
cv2.setMouseCallback('watershed', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()