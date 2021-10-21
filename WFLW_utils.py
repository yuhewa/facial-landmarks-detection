import numpy as np
import cv2
import os


WFLW_98_PTS_IDX = {
    "jaw": list(range(0, 33)),
    "left_eyebrow": list(range(33, 42)),
    "right_eyebrow": list(range(42, 51)),
    "nose": list(range(51, 60)),
    "nose_ver": list(range(51, 55)),
    "nose_hor": list(range(55, 60)),
    "left_eye": list(range(60, 68)) + [96],
    "right_eye": list(range(68, 76)) + [97],
    "left_eye_poly": list(range(60, 68)),
    "right_eye_poly": list(range(68, 76)),
    "mouth": list(range(76, 96)),
    "eyes": list(range(60, 68)) + [96] + list(range(68, 76)) + [97],
    "eyebrows": list(range(33, 42)) + list(range(42, 51)),
    "eyes_and_eyebrows": list(range(33, 42))
    + list(range(42, 51))
    + list(range(60, 68))
    + [96]
    + list(range(68, 76))
    + [97],
}


class LandmarksWFLW():
    def __init__(self, imgs_root_dir, anno_path):
        self.imgs_landmarks = {}
        self.imgs_root_dir = imgs_root_dir
        self.imgs_dirs = []
        self.imgs_path = []

        # get anno file
        with open(anno_path) as f:
            lines = f.readlines()  # len(lines)=7500, means 7500 faces
            for line in lines:
                line = line.split()
                img_name = line[206].split("/")[1]
                ### if != '1', means there are multi-face in a image, so concat the face anno
                if self.imgs_landmarks.get(img_name, "1") != "1":
                    self.imgs_landmarks[img_name] += list(
                        map(lambda x: int(float(x)), line[:196])
                    )
                else:
                    self.imgs_landmarks[img_name] = list(
                        map(lambda x: int(float(x)), line[:196])
                    )

        ###          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ###         planed to do: get the annotation and tanslate to point, not single value
        ###
        ###

        # 0_Para de_Parade_0_336.jpg can't find
        # get imgs dir path
        self.imgs_dirs = os.listdir(self.imgs_root_dir)
        self.imgs_dirs = list(
            sorted(self.imgs_dirs, key=lambda x: int(x.split("-")[0]))
        )

        for dir in self.imgs_dirs:
            one_dir_img = os.listdir(os.path.join(self.imgs_root_dir, dir))
            for img_name in one_dir_img:
                self.imgs_path.append([dir, img_name])

    def draw_major_points(self, img_path, img_landmarks):
        num_face = len(img_landmarks) // 196
        img = cv2.imread(img_path)
        landmarks = []
        print("number of face {}".format(num_face))
        for num in range(num_face):
            for i in range(196 * num, 196 * (num + 1), 2):
                landmarks.append((img_landmarks[i], img_landmarks[i + 1]))
        # total 98 point
        for num in range(num_face):
            for i in range(98 * num, 98 * (num + 1)):
                img = cv2.circle(
                    img, landmarks[i], radius=3, color=(0, 255, 0), thickness=-1
                )
        return img

    def show_landmarks(self):
        dir = self.imgs_path[8][0]
        img_name = self.imgs_path[8][1]
        img_path = os.path.join(self.imgs_root_dir, dir, img_name)
        # if the img has training annotation
        if self.imgs_landmarks.get(img_name, 0) != 0:
            img_landmarks = self.imgs_landmarks[img_name]
            img = self.draw_major_points(img_path, img_landmarks)

            extra_landmarks = self.coarse_interpolate(img_landmarks)
            img = self.draw_extra_points(img, extra_landmarks)

            cv2.imshow(img_path, img)
            cv2.cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _compute_extra_points(
        self, landmarks, points_group, extra_landmarks, is_circle
    ):
        for i in points_group[1:]:
            x = (landmarks[i][0] + landmarks[i - 1][0]) // 2
            y = (landmarks[i][1] + landmarks[i - 1][1]) // 2
            extra_landmarks.append((x, y))
        if is_circle == True:
            x = (landmarks[points_group[-1]][0] + landmarks[points_group[0]][0]) // 2
            y = (landmarks[points_group[-1]][1] + landmarks[points_group[0]][1]) // 2
            extra_landmarks.append((x, y))

    def coarse_interpolate(self, img_landmarks):
        num_face = len(img_landmarks) // 196
        landmarks = []
        for num in range(num_face):
            for i in range(196 * num, 196 * (num + 1), 2):
                landmarks.append((img_landmarks[i], img_landmarks[i + 1]))

        extra_landmarks = []

        for num in range(num_face):
            for i in range(num, 98 * (num + 1)):
                self._compute_extra_points(
                    landmarks, WFLW_98_PTS_IDX["jaw"], extra_landmarks, is_circle=False
                )
                self._compute_extra_points(
                    landmarks,
                    WFLW_98_PTS_IDX["left_eyebrow"],
                    extra_landmarks,
                    is_circle=True,
                )
                self._compute_extra_points(
                    landmarks,
                    WFLW_98_PTS_IDX["right_eyebrow"],
                    extra_landmarks,
                    is_circle=True,
                )
                self._compute_extra_points(
                    landmarks,
                    WFLW_98_PTS_IDX["left_eye_poly"],
                    extra_landmarks,
                    is_circle=True,
                )
                self._compute_extra_points(
                    landmarks,
                    WFLW_98_PTS_IDX["right_eye_poly"],
                    extra_landmarks,
                    is_circle=True,
                )
                self._compute_extra_points(
                    landmarks,
                    WFLW_98_PTS_IDX["nose_ver"],
                    extra_landmarks,
                    is_circle=False,
                )
                self._compute_extra_points(
                    landmarks,
                    WFLW_98_PTS_IDX["nose_hor"],
                    extra_landmarks,
                    is_circle=False,
                )
                self._compute_extra_points(
                    landmarks, WFLW_98_PTS_IDX["mouth"], extra_landmarks, is_circle=True
                )

        return extra_landmarks

    def draw_extra_points(self, img, extra_landmarks):
        for point in extra_landmarks:
            img = cv2.circle(img, point, radius=3, color=(0, 0, 255), thickness=-1)
        return img


if __name__ == "__main__":
    imgs_root_dir = "/home/gavin/facial_landmarks/datasets/WFLW/WFLW_images"
    anno_path = "/home/gavin/facial_landmarks/datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    data = LandmarksWFLW(imgs_root_dir, anno_path)
    data.show_landmarks()
    pass

## face around 33, 0~32
## left eyebrow 9, 33~42
## right eyebrow 9, 42~52
## nose ver:4 hor:5
## left eye out 8 52~61
## eight eye out 8 61~69
## lip out:12 in:5 5
## left eye center 1
## right eye center 1
