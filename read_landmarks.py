import numpy as np
import cv2
import os

img_root_dir = "/home/gavin/face_alignment/datasets/WFLW/WFLW_images"
anno_path = "/home/gavin/face_alignment/datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
img_landmarks = {}

### reading annotation ###
with open(anno_path) as f:
    lines = f.readlines() # len(lines)=7500, means 7500 image
    for line in lines:
        line = line.split()
        img_name = line[206].split("/")[1]
        img_landmarks[img_name] = list(map(lambda x: int(float(x)), line[:196]))
    # # 207 = 98*2 + 4 + 6 + 1
    # print(line[:196]) # points
    # print(line[196:200]) # bbox
    # print(line[200:206]) # attr
    # print(line[206]) # image
# print(img_landmarks.keys())


########### reading image ###
img_dirs = os.listdir(img_root_dir)
img_dirs = list(sorted(img_dirs, key=lambda x: int(x.split("-")[0]) ))
imgs = os.listdir(os.path.join(img_root_dir, img_dirs[0])) ## get one dir only
img_name = imgs[8] ## get one img only
img_path = os.path.join(img_root_dir,img_dirs[0],img_name)
img = cv2.imread(img_path)
landmarks_coorinate = img_landmarks[img_name]
print(landmarks_coorinate)
landmarks = []
for i in range(0,196,2):
    landmarks.append((landmarks_coorinate[i], landmarks_coorinate[i+1])) 
print(landmarks)

## face around 33, 0~32 
## left eyebrow 9, 33~42
## right eyebrow 9, 42~52 
## nose ver:4 hor:5 
## left eye out 8 52~61
## eight eye out 8 61~69
## lip out:12 in:5 5
## left eye center 1
## right eye center 1



# #### draw line
# # face round 33 points
# for i in range(0,32*2,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
# face round 33 points
for i in range(32):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)

# # left eyebrow 9 points
# for i in range(33*2,33*2+9*2-2,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(33,33+8):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)
img = cv2.line(img, landmarks[33], landmarks[33+8], (0, 0, 255), 3)

# # right eyebrow 9 points
# for i in range(42*2,42*2+9*2-2,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(42,42+8):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)
img = cv2.line(img, landmarks[42+8], landmarks[42], (0, 0, 255), 3)

# # nose 4 vertical points
# for i in range(51*2,51*2+4*2-2,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(51,51+3):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)

# # nose 4 horizontal points
# for i in range(55*2,55*2+4*2,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(55,55+4):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)

# # left eye out 8 points
# for i in range(60*2,60*2+9*2-4,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(60,60+7):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)
img = cv2.line(img, landmarks[60+7], landmarks[60], (0, 0, 255), 3)

# # right eye out 8 points
# for i in range(68*2,68*2+9*2-4,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(68,68+7):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)
img = cv2.line(img, landmarks[68+7], landmarks[68], (0, 0, 255), 3)


# # lip out 12 points
# for i in range(76*2,76*2+12*2,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(76,76+12):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)

# # lip in 10 points
# for i in range(86*2,86*2+10*2-2,2):
#     img = cv2.line(img, (landmarks[i],landmarks[i+1]), (landmarks[i+2],landmarks[i+3]), (0, 0, 255), 3)
for i in range(86,86+8):
    img = cv2.line(img, landmarks[i], landmarks[i+1], (0, 0, 255), 3)

#### draw point
# total 98 point
# for i in range(0,98*2,2):
#     img = cv2.circle(img, landmarks[i], radius=4, color=(0, 255, 0), thickness=-1)
for i in range(98):
    img = cv2.circle(img, landmarks[i], radius=4, color=(0, 255, 0), thickness=-1)

cv2.imshow('My Image', img)
cv2.cv2.waitKey(0)
cv2.destroyAllWindows()
    