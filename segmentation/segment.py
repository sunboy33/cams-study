import segmentation_models_pytorch as smp
import torch



import warnings

warnings.filterwarnings("ignore")



num_classes = 21
net = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
checkpoint = torch.load("checkpoint.pt")
net.load_state_dict(checkpoint)
net.to(device)


with open("../detection/val.txt","r") as f:
    lines = f.read().splitlines()

img_path = "../VOCdevkit/VOC2012/JPEGImages"





