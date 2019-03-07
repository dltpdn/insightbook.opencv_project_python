import cv2

img = cv2.imread('../img/blank_500.jpg')  # OpenCV로 이미지 읽기
print( type(img))                   # img의 데이타 타입
print(img.ndim)     # 배열의 차원 수 
print( img.shape)   # 각 차원의 크기
print(img.size)     # 전체 요소의 갯수
print( img.dtype)   # 데이타 타입
print(img.itemsize) # 각 요소의 바이트 크기