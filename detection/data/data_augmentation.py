import random 
from torchvision import transforms as T

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = T.ToTensor()(image)
        return image, target
    
class Normalize():
    def __call__(self, image, target):
        image = T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])(image)
        return image, target