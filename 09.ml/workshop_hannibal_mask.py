import cv2
import numpy as np

# 마스크 이미지 읽기 
face_mask = cv2.imread('../img/mask_hannibal.png')
h_mask, w_mask = face_mask.shape[:2]
# 얼굴 검출기 생성
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 얼굴 영역 검출
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        if h > 0 and w > 0:
        		# 마스크 위치 보정
            x = int(x + 0.1*w)
            y = int(y + 0.4*h)
            w = int(0.8 * w)
            h = int(0.8 * h)

            frame_roi = frame[y:y+h, x:x+w]
            # 마스크 이미지를 얼굴 크기에 맞게 조정 
            face_mask_small = cv2.resize(face_mask, (w, h), \
                                interpolation=cv2.INTER_AREA)
			# 마스크 이미지 합성
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small,\
                                         mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
            frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)

    cv2.imshow('Hanibal Mask', frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
