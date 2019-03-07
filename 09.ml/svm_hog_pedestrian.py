import cv2

# default 디덱터를 위한 HOG 객체 생성 및 설정--- ①
hogdef = cv2.HOGDescriptor()
hogdef.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# dailer 디덱터를 위한 HOG 객체 생성 및 설정--- ②
hogdaim  = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
hogdaim.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

cap = cv2.VideoCapture('../img/walking.avi')
mode = True  # 모드 변환을 위한 플래그 변수 
print('Toggle Space-bar to change mode.')
while cap.isOpened():
    ret, img = cap.read()
    if ret :
        if mode:
            # default 디텍터로 보행자 검출 --- ③
            found, _ = hogdef.detectMultiScale(img)
            for (x,y,w,h) in found:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255))
        else:
            # daimler 디텍터로 보행자 검출 --- ④
            found, _ = hogdaim.detectMultiScale(img)
            for (x,y,w,h) in found:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0))
        cv2.putText(img, 'Detector:%s'%('Default' if mode else 'Daimler'), \
                        (10,50 ), cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),1)
        cv2.imshow('frame', img)
        key = cv2.waitKey(1) 
        if key == 27:
            break
        elif key == ord(' '):
            mode = not mode
    else:
        break
cap.release()
cv2.destroyAllWindows()