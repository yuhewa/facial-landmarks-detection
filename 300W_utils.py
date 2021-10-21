import numpy as np
import cv2
import os
from scipy.io import loadmat

class Landmarks300W():
    def __init__(self, root_dir, bbox_path) -> None:
        self.root_dir = root_dir
        self.img_dirs = img_dirs = os.listdir(root_dir)
        self.imgs_path = []
        self.anno_path = []

        
        for dir in self.img_dirs:
            all_path = os.listdir(os.path.join(root_dir, dir)) 
            all_path.sort(key=lambda x: x.split(".")[1])
            self.imgs_path.append(sorted( all_path[:300], key=lambda x: int(x.split(".")[0][-3:])))
            self.anno_path.append(sorted( all_path[300:], key=lambda x: int(x.split(".")[0][-3:])))

    def show_landmarks(self):
        img_path = os.path.join(self.root_dir, self.img_dirs[0], self.imgs_path[0][0])
        img = cv2.imread(img_path)

        ladnmarks = self.anno_path

        cv2.imshow(img_path, img)
        cv2.cv2.waitKey(0)
        cv2.destroyAllWindows()

    # template
    def get_bbox(self):
        f = loadmat(os.path.join(self.root_dir, "Bounding Boxes/bounding_boxes_ibug.mat") )
        # print(f.keys())
        keys = ['__header__', '__version__', '__globals__', 'bounding_boxes']
        for bbox in f[keys[3]]:
            print(bbox[0])
            print(bbox[1])


if __name__ == "__main__":
    root_dir = "/home/gavin/facial_landmarks/datasets/300W_related/300W"
    bbox_path = "/home/gavin/facial_landmarks/datasets/300W_related/Bounding Boxes"
    data = Landmarks300W(root_dir, bbox_path)
    data.show_landmarks()
    



    