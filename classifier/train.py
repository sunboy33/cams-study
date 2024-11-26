from torchvision.datasets import ImageFolder




img_path = "~/datasets/animal10/train_data"
val_path = "~/datasets/animal10/val_data"
transform  = None

train_datasets = ImageFolder(root=img_path,transform=transform)
val_datasets = ImageFolder(root=val_path,transform=transform)

print(train_datasets)
print(val_datasets)