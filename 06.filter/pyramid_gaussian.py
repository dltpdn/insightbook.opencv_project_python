import cv2

img = cv2.imread('../img/girl.jpg')

# 가우시안 이미지 피라미드 축소
smaller = cv2.pyrDown(img) # img x 1/4
# 가우시안 이미지 피라미드 확대
bigger = cv2.pyrUp(img) # img x 4

# 결과 출력
cv2.imshow('img', img)
cv2.imshow('pyrDown', smaller)
cv2.imshow('pyrUp', bigger)
cv2.waitKey(0)
cv2.destroyAllWindows()
