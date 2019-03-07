import cv2

video_file = "../img/big_buck.avi" # 동영상 파일 경로

cap = cv2.VideoCapture(video_file) # 동영상 캡쳐 객체 생성
if cap.isOpened():                 # 캡쳐 객체 초기화 확인
    fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
    delay = int(1000/fps)
    print("FPS: %f, Delay: %dms" %(fps, delay))

    while True:
        ret, img = cap.read()      # 다음 프레임 읽기
        if ret:                     # 프레임 읽기 정상
            cv2.imshow(video_file, img) # 화면에 표시
            cv2.waitKey(delay)            # fps에 맞게 시간 지연
        else:
            break                   # 다음 프레임 읽을 수 없슴, 재생 완료
else:
    print("can't open video.")      # 캡쳐 객체 초기화 실패
cap.release()                       # 캡쳐 자원 반납
cv2.destroyAllWindows()