import os
import warnings

import torch
import numpy as np
import torchvision

from torchvision import transforms
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


warnings.filterwarnings("ignore")


transforms_val = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


def plot_detect_cams(img_path, name):
    device = torch.device("cuda:0")
    num_classes = 21
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load("/home/sunboy/CAMs/detection/checkpoint/faster_RCNN_best.pth")

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    target_layer = [model.backbone.fpn.layer_blocks[-1]]
    image = Image.open(os.path.join(img_path, name + ".jpg")).convert("RGB")
    rgb_img = np.array(image) / 255.0
    img_tensor = transforms_val(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    targets = [FasterRCNNBoxScoreTarget(labels=predictions["labels"], bounding_boxes=predictions["boxes"])]
    with GradCAM(model=model, target_layers=target_layer,reshape_transform=None) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(np.uint8(grayscale_cam * 255)).resize(image.size)
    Image.fromarray(cam_image).resize(image.size).save('1.png')



    


if __name__ == "__main__":
    img_path = "VOCdevkit/VOC2012/JPEGImages"
    val_txt = "val.txt"
    plot_detect_cams(img_path, "2007_000027")
