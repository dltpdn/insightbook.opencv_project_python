import cv2

ksize = 30              # 블러 처리에 사용할 커널 크기
win_title = 'mosaic'    # 창 제목
img = cv2.imread('../img/taekwonv1.jpg')    # 이미지 읽기

while True:
    x,y,w,h = cv2.selectROI(win_title, img, False) # 관심영역 선택
    if w > 0 and h > 0:         # 폭과 높이가 음수이면 드래그 방향이 옳음 
        roi = img[y:y+h, x:x+w]   # 관심영역 지정
        roi = cv2.blur(roi, (ksize, ksize)) # 블러(모자이크) 처리
        img[y:y+h, x:x+w] = roi   # 원본 이미지에 적용
        cv2.imshow(win_title, img)
    else:
        break
cv2.destroyAllWindows()