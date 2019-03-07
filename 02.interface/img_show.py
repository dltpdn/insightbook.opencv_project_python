'''
This code is provided for the book <OpenCV Project using Python> published by Insight book Inc.   
This code is released under the MIT license, and is available on GitHub site below

이 코드는 인사이트출판사에서 출판된 책 <파이썬으로 만드는 OpenCV 프로젝트>를 위해 제공합니다.
이 코드는 MIT 라이센스를 따르고 아래의 GitHub 주소에서도 받을 수 있습니다.

GitHub : https://github.com/dltpdn/book_opencv_prject_using_python
Author : Lee Sewoo(이세우, dltpdn@gmail.com)
'''
import cv2

img_file = "../img/girl.jpg" # 표시할 이미지 경로            ---①
img = cv2.imread(img_file)  # 이미지를 읽어서 img 변수에 할당 ---②

if img is not None:
  cv2.imshow('IMG', img)   # 읽은 이미지를 화면에 표시      --- ③
  cv2.waitKey()           # 키가 입력될 때 까지 대기      --- ④
  cv2.destroyAllWindows()  # 창 모두 닫기            --- ⑤
else:
    print('No image file.')
    