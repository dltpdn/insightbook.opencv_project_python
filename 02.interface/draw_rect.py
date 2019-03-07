import cv2

img = cv2.imread('../img/blank_500.jpg')

# 좌상, 우하 좌표로 사각형 그리기
cv2.rectangle(img, (50, 50), (150, 150), (255,0,0) )        
# 우하, 좌상 좌표로 사각형 그리기
cv2.rectangle(img, (300, 300), (100, 100), (0,255,0), 10 )  
# 우상, 좌하 좌표로 사각형 채워 그리기 ---①
cv2.rectangle(img, (450, 200), (200, 450), (0,0,255), -1 )  

cv2.imshow('rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()