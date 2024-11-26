import os
import xml.etree.ElementTree as ET
import pandas as pd

segment_val_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
imgs_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/JPEGImages"
xml_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/Annotations"

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
           'sheep', 'sofa', 'train', 'tvmonitor']


with open(segment_val_path,"r") as f:
    val_imgs = f.read().splitlines()

all_imgs = os.listdir(imgs_path)

train_imgs = [img[:-4] for img in all_imgs if img[:-4] not in val_imgs]

# print(len(val_imgs)) # 1449

# print(len(all_imgs)) # 17152

# print(len(train_imgs)) # 15676
labels = []
img_names = []
for img in train_imgs:
    xml_file = os.path.join(xml_path,img+".xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall("object")
    label = [0] * 20
    for obj in objects:
        if obj.find("difficult").text == "1":
            continue
        name = obj.find("name").text
        index = classes.index(name)
        label[index] = 1
    img_names.append(img)
    labels.append(label)


file = "data/trainsets.xlsx"
write = pd.ExcelWriter(file,engine='openpyxl',mode="w")
df = pd.DataFrame(labels)
df.index = img_names
df.to_excel(excel_writer=write,sheet_name="train",header=True)
write._save()


labels = []
img_names = []
for img in val_imgs:
    xml_file = os.path.join(xml_path,img+".xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall("object")
    label = [0] * 20
    for obj in objects:
        if obj.find("difficult").text == "1":
            continue
        name = obj.find("name").text
        index = classes.index(name)
        label[index] = 1
    img_names.append(img)
    labels.append(label)

file = "data/valsets.xlsx"
write = pd.ExcelWriter(file,engine='openpyxl',mode="w")
df = pd.DataFrame(labels)
df.index = img_names
df.to_excel(excel_writer=write,sheet_name="val",header=True)
write._save()





