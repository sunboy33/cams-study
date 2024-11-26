import torch
from segmentation.utils import StreamSegMetrics
from segmentation.train import *
import segmentation.network as network
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


num_classes = 21

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = network.deeplabv3plus_resnet50(output_stride=16, pretrained_backbone=False)
checkpoint = torch.load("./segmentation/checkpoint.pt")
model.load_state_dict(checkpoint)
model.to(device)


metrics = StreamSegMetrics(n_classes=21)

with open("./class_names.txt", "r") as f:
    lines = f.read().splitlines()


def validate(model, loader, device, metrics):
    metrics.reset()
    model.eval()
    with torch.no_grad():
        for sample in loader:
            images = sample[0].to(device, dtype=torch.float32)
            labels = sample[1].to(device, dtype=torch.long)
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
        score = metrics.get_results()
    df = pd.DataFrame(list(score["Class IoU"].items()), columns=["class", "iou"])
    print(score["Class IoU"])
    df.to_excel("seg_metrics.xlsx")


with open("./val.txt", "r") as file:
    names = file.read().splitlines()

img_path = "./VOCdevkit/VOC2012/JPEGImages/"
label_path = "./VOCdevkit/VOC2012/SegmentationClass/"
val_transform = et.ExtCompose(
    [
        et.ExtResize((512, 512)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

from PIL import Image
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
import matplotlib.pyplot as plt


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


def segment():
    save_path = "./segments/"
    model.eval()
    with torch.no_grad():
        n = 0
        for name in names:
            img = os.path.join(img_path, name + ".jpg")
            label = os.path.join(label_path, name + ".png")
            image = Image.open(img).convert("RGB")
            label = Image.open(label).resize((512, 512))
            img_tensor, _ = val_transform(image, label)
            img_tensor1 = img_tensor.unsqueeze(0).to(device)
            output = model(img_tensor1)
            preds = output.detach().max(dim=1)[1].cpu().numpy()[0]
            d_tensor = denormalize(
                img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            preds = np.array(Image.fromarray(np.uint8(preds)).resize(image.size))
            d_arr = np.uint8(d_tensor.numpy().transpose(1, 2, 0) * 255)
            d_arr = np.array(Image.fromarray(d_arr).resize(image.size))
            num_classes = 21
            colors = [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [255, 100, 0],
                [10, 100, 255],
            ]

            color_mask = np.zeros_like(d_arr)

            for class_id in range(1, num_classes):
                class_mask = preds == class_id
                color_mask[class_mask] = colors[class_id]
            alpha = 0.4
            img_with_mask = cv2.addWeighted(d_arr, 1 - alpha, color_mask, alpha, 0)
            w, h = image.size
            plt.figure(figsize=(w / 100, h / 100))
            plt.imshow(cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(f"{save_path}{name}.png")
            n += 1
            print(f"正在处理第{n}张图片...done!")


from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam import GradCAM,EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(lines)}


def plot_segment_cams(img_path, name):
    model.eval()
    target_layers = [model.backbone.layer4[-1]]
    img = os.path.join(img_path, name + ".jpg")
    label = os.path.join(label_path, name + ".png")
    image = Image.open(img).convert("RGB")
    label = Image.open(label)
    img_tensor, _ = val_transform(image, label)
    img_tensor1 = img_tensor.unsqueeze(0).to(device)
    output = model(img_tensor1)
    preds = output.detach().max(dim=1)[1].cpu().numpy()[0]
    idxs = np.unique(preds)
    targets = []
    for idx in idxs:
        if idx == 0:
            continue
        mask_float = np.float32(preds == idx)
        targets.append(SemanticSegmentationTarget(idx, mask_float))
    if len(targets) == 0:
        mask_float = np.float32(preds == 1)
        targets.append(SemanticSegmentationTarget(1, mask_float))
    d_tensor = denormalize(
        img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    rgb_img = d_tensor.numpy().transpose(1, 2, 0)

    repeated_tensor = img_tensor1.squeeze(0)[None, :].repeat(len(targets), 1, 1, 1)
    with GradCAM(model=model, target_layers=target_layers) as cam:
        batch_results = cam(input_tensor=repeated_tensor, targets=targets)
        gray = np.zeros((batch_results.shape[1], batch_results.shape[1]))
        for grayscale_cam in batch_results:
            # gray = np.clip(gray + grayscale_cam, 0, 1)
            gray = 0.5*gray + grayscale_cam*0.5
        cam_image = show_cam_on_image(rgb_img, gray, use_rgb=True)

    return Image.fromarray(np.uint8(grayscale_cam * 255)).resize(
        image.size
    ), Image.fromarray(cam_image).resize(image.size)


if __name__ == "__main__":
    pass
    # validate(model, dl_val, device, metrics)
    # segment()
    # plot_cams()
    # print(np.array(Image.open("../VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg")).shape)
