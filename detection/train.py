import os

import torch
import torchvision

from torchkeras import KerasModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data import VOC
from data.data_augmentation import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class StepRunner:
    def __init__(
        self,
        net,
        loss_fn,
        accelerator,
        stage="train",
        metrics_dict=None,
        optimizer=None,
        lr_scheduler=None,
    ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = (
            net,
            loss_fn,
            metrics_dict,
            stage,
        )
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
        if self.stage == "train":
            self.net.train()
        else:
            self.net.train()

    def __call__(self, batch):
        features, labels = batch

        # loss
        loss_dict = self.net(features, labels)
        loss = sum(loss_dict.values())

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # all_preds = self.accelerator.gather(preds)
        # all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        # losses
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics["lr"] = self.optimizer.state_dict()["param_groups"][0][
                    "lr"
                ]
            else:
                step_metrics["lr"] = 0.0
        return step_losses, step_metrics


KerasModel.StepRunner = StepRunner


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    transforms_train = Compose([ToTensor(), RandomHorizontalFlip(0.5), Normalize()])
    transforms_val = Compose([ToTensor(), Normalize()])

    img_path = "../VOCdevkit/VOC2012/JPEGImages"
    xml_path = "../VOCdevkit/VOC2012/Annotations"
    train_txt_path = "train.txt"
    val_txt_path = "val.txt"
    ds_train = VOC(img_path, xml_path, train_txt_path, transforms=transforms_train)
    ds_val = VOC(img_path, xml_path, val_txt_path, transforms=transforms_val)

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn
    )

    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    num_classes = 21
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    params = [
        {"params": model.backbone.parameters(), "lr": 0.0001},
        {"params": model.roi_heads.parameters(), "lr": 0.001},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    keras_model = KerasModel(
        model,
        loss_fn=None,
        metrics_dict=None,
        optimizer=optimizer,
        lr_scheduler=None,
    )

    keras_model.fit(
        train_data=dl_train,
        val_data=dl_val,
        epochs=50,
        patience=8,
        monitor="val_loss",
        mode="min",
        ckpt_path="checkpoint/faster_RCNN_best.pth",
        plot=True,
    )


if __name__ == "__main__":
    main()
