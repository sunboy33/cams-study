import xml.etree.ElementTree as ET
from PIL import Image
import torch
import os

import warnings
warnings.filterwarnings('ignore')

with open('../class_names.txt','r') as file:
    class_names = file.read().splitlines()
# print(class_names)

class VOC(torch.utils.data.Dataset):
    def __init__(self, img_path, xml_path, txt_path, class_names = class_names,transforms = None):
        with open(txt_path,'r') as file:
            self.lines = file.read().splitlines()
        self.img_path = img_path
        self.xml_path = xml_path
        self.class_names = class_names
        self.transforms = transforms
        
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path,self.lines[idx] + '.jpg')).convert('RGB')
        tree = ET.parse(os.path.join(self.xml_path,self.lines[idx] + '.xml'))
        root = tree.getroot()
        objects = root.findall('object')
        
        target = {}
        boxes = []
        labels = []
        scores = []
        for obj in objects:
            if obj.find('difficult') == "1":
                continue
            label = self.class_names.index(obj.find('name').text)
            xmin = float(obj.find('bndbox').find('xmin').text)
            ymin = float(obj.find('bndbox').find('ymin').text)
            xmax = float(obj.find('bndbox').find('xmax').text)
            ymax = float(obj.find('bndbox').find('ymax').text)
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(label)
            scores.append(1)
        target['boxes'] = torch.tensor(boxes)
        target['labels'] = torch.tensor(labels)
        target['scores'] = torch.tensor(scores)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
  
        return img, target

    def __len__(self):
        return len(self.lines)
    

    