import cv2
import numpy as np

# 얼굴과 눈동자 검출기 생성
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

# 렌즈 왜곡 효과 함수
def distortedMap(rows, cols, type=0):
    map_y, map_x = np.indices((rows, cols), dtype=np.float32)
    # 렌즈 효과
    ## 렌즈 효과, 중심점 이동
    map_lenz_x = (2*map_x - cols)/cols
    map_lenz_y = (2*map_y - rows)/rows
    ## 렌즈 효과, 극좌표 변환
    r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)
    if type==0:
    ## 볼록 렌즈 효과 매핑 좌표 연산
        r[r< 1] = r[r<1] **3  
    else:
    ## 오목 렌즈 효과 매핑 좌표 연산
        r[r< 1] = r[r<1] **0.5
    ## 렌즈 효과, 직교 좌표 복원
    mapx, mapy = cv2.polarToCart(r, theta)
    ## 렌즈 효과, 좌상단 좌표 복원
    mapx = ((mapx + 1)*cols)/2
    mapy = ((mapy + 1)*rows)/2
    return (mapx, mapy)

# 얼굴 검출 함수
def findFaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    face_coords = []
    for (x,y,w,h) in faces:
        face_coords.append((x, y, w, h))
    return face_coords
# 눈 검출 함수
def findEyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    eyes_coords = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray )
        for(ex,ey,ew,eh) in eyes:
            eyes_coords.append((ex+x,ey+y,ew,eh))
    return eyes_coords


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

while True:
    ret, frame = cap.read()
    img1 = frame.copy()
    img2 = frame.copy()
    # 얼굴 검출해서 오목/볼록 렌즈 효과로 왜곡 적용
    faces = findFaces(frame)
    for face in faces:
        x,y,w,h = face
        mapx, mapy = distortedMap(w,h, 1)
        roi = img1[y:y+h, x:x+w]
        convex = cv2.remap(roi,mapx,mapy,cv2.INTER_LINEAR)
        img1[y:y+h, x:x+w] = convex
    # 눈 영역 검출해서 볼록 렌즈 효과로 왜곡 적용
    eyes = findEyes(frame)
    for eye in eyes :
        x,y,w,h = eye
        mapx, mapy = distortedMap(w,h)
        roi = img2[y:y+h, x:x+w]
        convex = cv2.remap(roi,mapx,mapy,cv2.INTER_LINEAR)
        img2[y:y+h, x:x+w] = convex
    # 하나의 이미지로 병합해서 출력
    merged = np.hstack((frame, img1, img2))
    cv2.imshow('Face Distortion', merged)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
        
