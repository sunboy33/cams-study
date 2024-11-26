import pandas as pd
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import average_precision_score
import numpy as np
import cv2

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from train import MultiLabelClassifier
from utils import *


classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
           'sheep', 'sofa', 'train', 'tvmonitor']


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

img_root = "/home/sunboy/CAMs/VOCdevkit/VOC2012/JPEGImages/"
model = MultiLabelClassifier(20).to(device)
checkpoint = torch.load("/home/sunboy/CAMs/multi_label/best.pth", weights_only=True,map_location="cpu")
model.load_state_dict(checkpoint)



def parse(l):
    l = np.array(l)
    results = []
    targets = []
    indices = np.where(l == 1)[0].tolist()
    if len(indices) == 0:
        result = "None"
        targets.append(ClassifierOutputTarget(0))
    else:
        for index in indices:
            results.append(classes[index])
            targets.append(ClassifierOutputTarget(index))
        result = " ".join(results)
    return result, targets


def inference():
    model.eval()
    df = pd.read_excel("data/valsets.xlsx",index_col=0)
    pres = []
    scores = []
    imgs = df.index
    with torch.no_grad():
        i = 0
        for line in imgs:
            image = Image.open(os.path.join(img_root, line + ".jpg")).convert("RGB")
            img_tensor = val_transform(image).unsqueeze(0).to(device)
            label = df.loc[line].values.tolist()
            outputs = model(img_tensor)
            score = outputs.squeeze(0).float().cpu().numpy().tolist()
            scores.append(score)
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            pre = outputs.squeeze(0).long().cpu().numpy().tolist()
            pres.append(pre)
            i += 1
            print(f"预测第{i}张图片...done!")
    file = "data/valPredicts.xlsx"
    write = pd.ExcelWriter(file,engine='openpyxl',mode="w")
    df1 = pd.DataFrame(pres)
    df1.index = imgs
    df1.to_excel(excel_writer=write,sheet_name="predicts",header=True)
    df2 = pd.DataFrame(scores)
    df2.index = imgs
    df2.to_excel(excel_writer=write,sheet_name="scores",header=True)
    write._save()


def cal_map():
    df1 = pd.read_excel("data/valsets.xlsx",sheet_name="val",index_col=0)
    df2 = pd.read_excel("data/valPredicts.xlsx",sheet_name="scores",index_col=0)
    # print(df1[0])
    # print(df2[0])
    aps = []
    for col in df2.columns:
        ap = average_precision_score(y_true=df1[col].values,y_score=df2[col].values)
        aps.append(ap)
    ap_dict = {cls: ap for cls, ap in zip(classes,aps)}
    ap_dict["mean"] = sum(aps)/len(aps)
    pd.DataFrame(list(ap_dict.items())).to_excel("mAP.xlsx")






cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}


def plot_multi_label_cams():
    model.eval()
    df = pd.read_excel("data/valsets.xlsx",index_col=0)
    imgs = df.index
    i = 0
    for img in imgs:
        image = Image.open(os.path.join(img_root, img + ".jpg")).convert("RGB")
        img_tensor = val_transform(image).unsqueeze(0).to(device)
        outputs = model(img_tensor)
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        pre = outputs.squeeze(0).long().cpu().numpy().tolist()
        pre, targets = parse(pre)
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        target_layers = [model.resnet.layer4[-1]]
        repeated_tensor = img_tensor.squeeze(0)[None, :].repeat(len(targets), 1, 1, 1)
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grays = cam(input_tensor=repeated_tensor, targets=targets)
            results = []
            for gray in grays:
                cam_image = show_cam_on_image(rgb_img, gray, use_rgb=True)
                cam_image = cv2.resize(cam_image,image.size)
                results.append(cam_image)
        cams = np.hstack(results)
        Image.fromarray(cams).save(f'cams/{img}.png')
        i += 1
        print(f"处理第{i}张图片...done!")

if __name__ == "__main__":
    # cal_map()
    plot_multi_label_cams()
