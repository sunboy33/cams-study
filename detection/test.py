import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchkeras.plots import plot_detection
from tqdm import tqdm
from data.voc import class_names
from data.data_augmentation import *

with open("/home/sunboy/CAMs/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r") as f:
        lines = f.read().splitlines()

root_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/JPEGImages/"

transforms_val = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

device = torch.device("cuda:0")
num_classes = 21
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
checkpoint = torch.load("checkpoint/faster_RCNN_best.pth")
model.load_state_dict(checkpoint)
model.to(device)
model.eval()


bar = tqdm(total=len(lines))

def main():
    with torch.no_grad():
        for line in lines[:100]:
            image = Image.open(os.path.join(root_path,line + '.jpg')).convert("RGB")
            img_tensor = transforms_val(image).unsqueeze(0).to(device)
            output = model(img_tensor)[0]
            detect_img = plot_detection(image, output, class_names, min_score=0.7)
            save_path = os.path.join("detect_imgs/", line + ".jpg")
            detect_img.save(save_path)
            bar.update()

def get_detect_txts():
     with torch.no_grad():
          for line in lines:
            image = Image.open(os.path.join(root_path,line + '.jpg')).convert("RGB")
            img_tensor = transforms_val(image).unsqueeze(0).to(device)
            output = model(img_tensor)[0]
            boxes = output["boxes"].cpu()
            labels = output["labels"].cpu()
            scores = output["scores"].cpu()
            f = open(f"mAP-master/input/detection-results/{line}.txt","w")
            for i in range(len(boxes)):
                 f.write(class_names[labels[i]]+ " "+ str(scores[i].item())+ " "+ str(boxes[i][0].item()) + " " + str(boxes[i][1].item()) + " " + str(boxes[i][2].item()) + " " + str(boxes[i][3].item()) + "\n")
            f.close()
            bar.update()

import xml.etree.ElementTree as ET
def show_with_tagets(img_path,xml_path):
    image = Image.open(img_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    
    target = {}
    boxes = []
    labels = []
    scores = []
    for obj in objects:
        label = class_names.index(obj.find('name').text)
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        boxes.append([xmin,ymin,xmax,ymax])
        labels.append(label)
        scores.append(1)
    target['boxes'] = torch.tensor(boxes)
    target['labels'] = torch.tensor(labels)
    target['scores'] = torch.tensor(scores)
    detect_img = plot_detection(image, target, class_names, min_score=0.7)
    save_path = os.path.join("detect.jpg")
    detect_img.save(save_path)
                 
     
        

if __name__ == "__main__":
    main()
    # get_detect_txts()
    # img_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/JPEGImages/2007_001586.jpg"
    # xml_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/Annotations/2007_001586.xml"
    # show_with_tagets(img_path,xml_path)
