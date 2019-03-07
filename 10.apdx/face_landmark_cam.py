import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성 --- ①
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
#cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 480)
#cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 320)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 영역 검출 --- ②
    faces = detector(gray)
    for rect in faces:
        # 얼굴 영역을 좌표로 변환 후 사각형 표시 --- ③
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
        # 얼굴 랜드마크 검출 --- ④
        shape = predictor(gray, rect)
        for i in range(68):
            # 부위별 좌표 추출 및 표시 --- ⑤
            part = shape.part(i)
            cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), -1)
#            cv2.putText(img, str(i), (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, 0.5,(255,255,255), 1, cv2.LINE_AA)
    
    cv2.imshow("face landmark", img)
    if cv2.waitKey(1)== 27:
        break
cap.release()



