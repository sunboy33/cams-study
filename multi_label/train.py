from torchvision import models
import torch.nn as nn
import torch
from utils import *


class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x


def train(model,train_loader,val_loader,optimizer,criterion,epochs,device):
    from tqdm import tqdm
    f = open("loss.txt","w")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        bar = tqdm(total=len(train_loader))
        for i, (imgs,labels) in enumerate(train_loader, 0):
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            bar.update()
        model.eval()
        val_total_loss = 0.0
        min_val_loss = 1000.0
        with torch.no_grad():
            for (imgs,labels) in val_loader:
                imgs, labels = imgs.to(device), labels.float().to(device)
                outputs = model(imgs)
                val_loss = criterion(outputs,labels)
                val_total_loss += val_loss.item()
        train_loss = running_loss / len(train_loader)
        val_loss = val_total_loss / len(val_loader)
        print('Epoch[%d/%d] train loss: %.6f  val loss: %.6f' % (epoch + 1 , 
                                                                 epochs,
                                                                 train_loss,
                                                                   val_loss))
        
        f.write(str(train_loss) + " "+ str(val_loss) + "\n")
        
        # if (epoch + 1) % 1 == 0:
        #     validate(model)
        
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(),'best.pth')
    f.close()


def main():
    torch.manual_seed(1991)
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )
    img_root = "/home/sunboy/CAMs/VOCdevkit/VOC2012/JPEGImages/"

    train_excel_file = "data/trainsets.xlsx"
    val_excel_flie = "data/valsets.xlsx"
    train_set = VOCClassify(img_root, train_excel_file, transform)
    dl_train = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=128, shuffle=True, num_workers=2
    )
    val_set = VOCClassify(img_root, val_excel_flie, val_transform)
    dl_val = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=64, shuffle=False, num_workers=2
    )

    model = MultiLabelClassifier(20).to(device)
    checkpoint = torch.load("/home/sunboy/CAMs/multi_label/checkpoint/best_valloss_0.0677.pth", weights_only=True,map_location="cpu")
    model.load_state_dict(checkpoint)

    # backbone_lr = 1e-3
    # classifier_lr = 1e-2
    # params = [
    #     {"params": model.resnet.conv1.parameters(), "lr": backbone_lr},
    #     {"params": model.resnet.bn1.parameters(), "lr": backbone_lr},
    #     {"params": model.resnet.layer1.parameters(), "lr": backbone_lr},
    #     {"params": model.resnet.layer2.parameters(), "lr": backbone_lr},
    #     {"params": model.resnet.layer3.parameters(), "lr": backbone_lr},
    #     {"params": model.resnet.layer4.parameters(), "lr": backbone_lr},
    #     {"params": model.resnet.fc.parameters(), "lr": classifier_lr},
    # ]
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001,momentum=0.9, weight_decay=1e-4)
    criterion = nn.BCELoss()
    train(model,dl_train,dl_val,optimizer,criterion,100,device)


if __name__ == "__main__":
    main()
