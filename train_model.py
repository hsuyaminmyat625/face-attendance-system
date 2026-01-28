# train_model.py
import cv2
import os
import numpy as np
import pickle

data_path = 'dataset'
faces = []
labels = []
label_map = {}
label_count = 0

# 各人物フォルダごとに学習
for person_name in os.listdir(data_path):
    person_path = os.path.join(data_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[label_count] = person_name  # 例: "田中_001"
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(label_count)

    label_count += 1

# NumPy配列に変換
faces = np.array(faces)
labels = np.array(labels)

# モデル学習
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)
model.save("face_model.yml")

# 名前とラベル対応表を保存
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ 学習完了！face_model.ymlとlabels.pklを保存しました。")

