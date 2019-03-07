import cv2
import numpy as np
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성 --- ①
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

img = cv2.imread("../img/man_face.jpg")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 얼굴 영역 검출 --- ②
rects = faces = detector(gray)

points = []
for rect in rects:
    # 랜드마크 검출 --- ③
    shape = predictor(gray, rect)
    for i in range(68):
        part = shape.part(i)
        points.append((part.x, part.y))
        

# 들로네 삼각 분할 객체 생성 --- ④
x,y,w,h = cv2.boundingRect(np.float32(points))
subdiv = cv2.Subdiv2D((x,y,x+w,y+h))
# 랜드마크 좌표 추가 --- ⑤
subdiv.insert(points)
# 들로네 삼각형 좌표 계산 --- ⑥
triangleList = subdiv.getTriangleList()
# 들로네 삼각형 그리기 --- ⑦
h, w = img.shape[:2]
cnt = 0
for t in triangleList :
    pts = t.reshape(-1,2).astype(np.int32)
    # 좌표 중에 이미지 영역을 벗어나는 것을 제외(음수 등) ---⑧
    if (pts < 0).sum() or (pts[:, 0] > w).sum() or (pts[:, 1] > h).sum():
        print(pts) 
        continue
    cv2.polylines(img, [pts], True, (255, 255,255), 1, cv2.LINE_AA)
    cnt+=1
print(cnt)


cv2.imshow("Delaunay",img)
cv2.waitKey(0)