import cv2
import numpy as np
import glob

# 영상 읽기 및 표시
img = cv2.imread('../img/pistol.jpg')
cv2.imshow('query', img)

# 비교할 영상들이 있는 경로 ---①
search_dir = '../img/101_ObjectCategories'

# 이미지를 16x16 크기의 평균 해쉬로 변환 ---②
def img2hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    avg = gray.mean()
    bi = 1 * (gray > avg)
    return bi

# 해밍거리 측정 함수 ---③
def hamming_distance(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a !=b).sum()
    return distance

# 권총 영상의 해쉬 구하기 ---④
query_hash = img2hash(img)

# 이미지 데이타 셋 디렉토리의 모든 영상 파일 경로 ---⑤
img_path = glob.glob(search_dir+'/**/*.jpg')
for path in img_path:
    # 데이타 셋 영상 한개 읽어서 표시 ---⑥
    img = cv2.imread(path)
    cv2.imshow('searching...', img)
    cv2.waitKey(5)
    # 데이타 셋 영상 한개의 해시  ---⑦
    a_hash = img2hash(img)
    # 해밍 거리 산출 ---⑧
    dst = hamming_distance(query_hash, a_hash)
    if dst/256 < 0.25: # 해밍거리 25% 이내만 출력 ---⑨
        print(path, dst/256)
        cv2.imshow(path, img)
cv2.destroyWindow('searching...')
cv2.waitKey(0)
cv2.destroyAllWindows()

