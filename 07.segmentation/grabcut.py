import cv2
import numpy as np

img = cv2.imread('../img/taekwonv1.jpg')
img_draw = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 마스크 생성
rect = [0,0,0,0]    # 사각형 영역 좌표 초기화
mode = cv2.GC_EVAL  # 그랩컷 초기 모드
# 배경 및 전경 모델 버퍼
bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)

# 마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param):
    global mouse_mode, rect, mask, mode
    if event == cv2.EVENT_LBUTTONDOWN : # 왼쪽 마우스 누름
        if flags <= 1: # 아무 키도 안 눌렀으면
            mode = cv2.GC_INIT_WITH_RECT # 드래그 시작, 사각형 모드 ---①
            rect[:2] = x, y # 시작 좌표 저장
    # 마우스가 움직이고 왼쪽 버튼이 눌러진 상태
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON :
        if mode == cv2.GC_INIT_WITH_RECT: # 드래그 진행 중 ---②
            img_temp = img.copy()
            # 드래그 사각형 화면에 표시
            cv2.rectangle(img_temp, (rect[0], rect[1]), (x, y), (0,255,0), 2)
            cv2.imshow('img', img_temp)
        elif flags > 1: # 키가 눌러진 상태
            mode = cv2.GC_INIT_WITH_MASK    # 마스크 모드 ---③
            if flags & cv2.EVENT_FLAG_CTRLKEY :# 컨트롤 키, 분명한 전경
                # 흰색 점 화면에 표시
                cv2.circle(img_draw,(x,y),3, (255,255,255),-1)
                # 마스크에 GC_FGD로 채우기      ---④
                cv2.circle(mask,(x,y),3, cv2.GC_FGD,-1)
            if flags & cv2.EVENT_FLAG_SHIFTKEY : # 쉬프트키, 분명한 배경
                # 검정색 점 화면에 표시
                cv2.circle(img_draw,(x,y),3, (0,0,0),-1)
                # 마스크에 GC_BGD로 채우기      ---⑤
                cv2.circle(mask,(x,y),3, cv2.GC_BGD,-1)
            cv2.imshow('img', img_draw) # 그려진 모습 화면에 출력
    elif event == cv2.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼 뗀 상태 ---⑥
        if mode == cv2.GC_INIT_WITH_RECT : # 사각형 그리기 종료
            rect[2:] =x, y # 사각형 마지막 좌표 수집
            # 사각형 그려서 화면에 출력 ---⑦
            cv2.rectangle(img_draw, (rect[0], rect[1]), (x, y), (255,0,0), 2)
            cv2.imshow('img', img_draw)
        # 그랩컷 적용 ---⑧
        cv2.grabCut(img, mask, tuple(rect), bgdmodel, fgdmodel, 1, mode)
        img2 = img.copy()
        # 마스크에 확실한 배경, 아마도 배경으로 표시된 영역을 0으로 채우기
        img2[(mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD)] = 0
        cv2.imshow('grabcut', img2) # 최종 결과 출력
        mode = cv2.GC_EVAL # 그랩컷 모드 리셋
# 초기 화면 출력 및 마우스 이벤트 등록
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
while True:    
    if cv2.waitKey(0) & 0xFF == 27 : # esc
        break
cv2.destroyAllWindows()