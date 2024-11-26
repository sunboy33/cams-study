import os
import shutil


root_path = "/home/sunboy/datasets/animal10/train_data"
val_path = "/home/sunboy/datasets/animal10/val_data"

cls = os.listdir(root_path)

d = {}
for c in cls:
    path = os.path.join(root_path,c)
    d[c] = len(os.listdir(path))

test_num_per_cls = 200

for c in cls:
    path = os.path.join(root_path,c)
    val = os.path.join(val_path,c)
    os.makedirs(val)
    img_names = os.listdir(path)
    for name in img_names[:test_num_per_cls]:
        img_path = os.path.join(path,name)
        dst_path = os.path.join(val,name)
        shutil.move(img_path,dst_path)
print("process done!")