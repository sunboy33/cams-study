import torch
from torchkeras import KerasModel
import segmentation.network as network
import os
from segmentation.utils import ext_transforms as et, VOCSegmentation
import torch.nn as nn

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
            self.net.eval()

    def __call__(self, batch):
        features, labels = batch[0], batch[1].long()
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        # losses
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics
        step_metrics = {
            self.stage + "_" + name: metric_fn(all_preds, all_labels).item()
            for name, metric_fn in self.metrics_dict.items()
        }

        if self.optimizer is not None and self.stage == "train":
            step_metrics["lr"] = self.optimizer.state_dict()["param_groups"][0]["lr"]

        return step_losses, step_metrics


KerasModel.StepRunner = StepRunner


img_size = 512

train_transform = et.ExtCompose(
    [
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(img_size, img_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transform = et.ExtCompose(
    [
        et.ExtResize(img_size),
        et.ExtCenterCrop(img_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_root = "/home/sunboy/CAMs/"
ds_train = VOCSegmentation(
    root=data_root,
    year="2012",
    image_set="train",
    download=False,
    transform=train_transform,
)
ds_val = VOCSegmentation(
    root=data_root,
    year="2012",
    image_set="val",
    download=False,
    transform=val_transform,
)

dl_train = torch.utils.data.DataLoader(
    ds_train, batch_size=16, shuffle=True, num_workers=2
)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=16)


def main():
    torch.manual_seed(1991)
    net = network.deeplabv3plus_resnet50(output_stride=16, pretrained_backbone=False)
    checkpoint = torch.load("best_deeplabv3plus_resnet50_voc_os16.pth")
    net.load_state_dict(checkpoint["model_state"])
    # for param in net.backbone.parameters():
    #     param.requires_grad = False
    optimizer = torch.optim.SGD(
        params=net.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=5, eta_min=0.0001
    )

    model = KerasModel(
        net,
        loss_fn=nn.CrossEntropyLoss(reduction="mean", ignore_index=255),
        metrics_dict=None,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    model.fit(
        train_data=dl_train,
        val_data=dl_val,
        epochs=100,
        ckpt_path="checkpoint.pt",
        patience=10,
        monitor="val_loss",
        mode="min",
        mixed_precision="no",
        plot=True,
    )


if __name__ == "__main__":
    main()
