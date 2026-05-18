import torchvision.transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD


def get_eval_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
