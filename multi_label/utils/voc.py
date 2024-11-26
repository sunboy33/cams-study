import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


xml_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/Annotations/"
root_path = "/home/sunboy/CAMs/VOCdevkit/VOC2012/ImageSets/Segmentation/"
data_root = "/home/sunboy/CAMs/VOCdevkit/VOC2012/ImageSets/Main/"

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
           'sheep', 'sofa', 'train', 'tvmonitor']


transform = transforms.Compose(
    [
        
        transforms.Resize((224, 224)),
        # transforms.RandomRotation(45),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomHorizontalFlip(0.1),
        # transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VOCClassify(torch.utils.data.Dataset):

    def __init__(self, img_root, excel_file, transform):
        super().__init__()
        self.img_root = img_root
        self.data = pd.read_excel(excel_file,index_col=0)
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.data.index[index]
        label = self.data.loc[img_name].values
        img_path = os.path.join(self.img_root, img_name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)
        label_tensor = torch.FloatTensor(label)
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_excel_file = "data/trainsets.xlsx"
    val_excel_flie = "data/valsets.xlsx"
    img_root = "/home/sunboy/CAMs/VOCdevkit/VOC2012/JPEGImages/"
    train_set = VOCClassify(img_root, train_excel_file, transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=64, shuffle=True, num_workers=2
    )
    val_set = VOCClassify(img_root, val_excel_flie, val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=4, shuffle=False, num_workers=2
    )

    for sampe in train_loader:
        print(sampe)
        break

