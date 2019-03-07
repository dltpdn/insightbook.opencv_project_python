import numpy as np, cv2

cap = cv2.VideoCapture('../img/walking.avi')
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)
# 추적 경로를 그리기 위한 랜덤 색상
color = np.random.randint(0,255,(200,3))
lines = None  #추적 선을 그릴 이미지 저장 변수
prevImg = None  # 이전 프레임 저장 변수
# calcOpticalFlowPyrLK 중지 요건 설정
termcriteria =  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임 경우
    if prevImg is None:
        prevImg = gray
        # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
        lines = np.zeros_like(frame)
        # 추적 시작을 위한 코너 검출  ---①
        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
    else:
        nextImg = gray
        # 옵티컬 플로우로 다음 프레임의 코너점  찾기 ---②
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, \
                                        prevPt, None, criteria=termcriteria)
        # 대응점이 있는 코너, 움직인 코너 선별 ---③
        prevMv = prevPt[status==1]
        nextMv = nextPt[status==1]
        for i,(p, n) in enumerate(zip(prevMv, nextMv)):
            px,py = p.ravel()
            nx,ny = n.ravel()
            # 이전 코너와 새로운 코너에 선그리기 ---④
            cv2.line(lines, (px, py), (nx,ny), color[i].tolist(), 2)
            # 새로운 코너에 점 그리기
            cv2.circle(img_draw, (nx,ny), 2, color[i].tolist(), -1)
        # 누적된 추적 선을 출력 이미지에 합성 ---⑤
        img_draw = cv2.add(img_draw, lines)
        # 다음 프레임을 위한 프레임과 코너점 이월
        prevImg = nextImg
        prevPt = nextMv.reshape(-1,1,2)

    cv2.imshow('OpticalFlow-LK', img_draw)
    key = cv2.waitKey(delay)
    if key == 27 : # Esc:종료
        break
    elif key == 8: # Backspace:추적 이력 지우기
        prevImg = None
cv2.destroyAllWindows()
cap.release()