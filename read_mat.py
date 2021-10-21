from scipy.io import loadmat


f = loadmat("/home/gavin/face_alignment/datasets/300W/Bounding Boxes/bounding_boxes_ibug.mat")

# print(f.keys())
keys = ['__header__', '__version__', '__globals__', 'bounding_boxes']
print(type(f[keys[3]]))

for bbox in f[keys[3]]:
    print(bbox[0])
    print(bbox[1])
    break

    