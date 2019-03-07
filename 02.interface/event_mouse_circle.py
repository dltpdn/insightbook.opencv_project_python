import cv2

title = 'mouse event'                   # 창 제목
img = cv2.imread('../img/blank_500.jpg') # 백색 이미지 읽기
cv2.imshow(title, img)                  # 백색 이미지 표시

def onMouse(event, x, y, flags, param): # 아무스 콜백 함수 구현 ---①
    print(event, x, y, )                # 파라미터 출력
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누름인 경우 ---②
        cv2.circle(img, (x,y), 30, (0,0,0), -1) # 지름 30 크기의 검은색 원을 해당 좌표에 그림
        cv2.imshow(title, img)          # 그려진 이미지를 다시 표시 ---③

cv2.setMouseCallback(title, onMouse)    # 마우스 콜백 함수를 GUI 윈도우에 등록 ---④

while True:
    if cv2.waitKey(0) & 0xFF == 27:     # esc로 종료
        break
cv2.destroyAllWindows()