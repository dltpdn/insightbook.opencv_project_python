import cv2
import numpy as np
import os, glob

# 변수 설정 --- ①
base_dir = './faces'
train_data, train_labels = [], []


dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]
print('Collecting train data set:')
for dir in dirs:
    # name_id 형식에서 id를 분리 ---②
    id = dir.split('_')[1]          
    files = glob.glob(dir+'/*.jpg')
    print('\t path:%s, %dfiles'%(dir, len(files)))
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # 이미지는 train_data, 아이디는 train_lables에 저장 ---③
        train_data.append(np.asarray(img, dtype=np.uint8))
        train_labels.append(int(id))

# NumPy 배열로 변환 ---④
train_data = np.asarray(train_data)
train_labels = np.int32(train_labels)

# LBP 얼굴인식기 생성 및 훈련 ---⑤
print('Starting LBP Model training...')
model = cv2.face.LBPHFaceRecognizer_create()
model.train(train_data, train_labels)
model.write('./faces/all_face.xml')
print("Model trained successfully!")
