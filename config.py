import albumentations as A
import torch.cuda
from albumentations.pytorch import ToTensorV2
import cv2

DEVICE, PIN_MEMORY = ["cuda", True] if torch.cuda.is_available() else ["cpu", False]

IMAGE_SIZE = 224
NUM_OF_CLASSES = 2
BATCH_SIZE = 32
NUM_WORKERS = 1
EPOCHS = 300
MODEL_FILENAME = "weights/resnet18.pth.tar"
# Параметры Лосса
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0005
FROZE_WEIGHTS = False
FEATURE_EXTRACT = False
# Визуализация
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (0, 0, 0)  # Black

classes = \
    ["None",
     "Apple"
     ]
train_transforms = A.Compose(
    [
        A.Resize(height=int(IMAGE_SIZE), width=int(IMAGE_SIZE), interpolation=cv2.INTER_LINEAR),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),

            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
        ToTensorV2(),
    ]
)
test_transforms = A.Compose(
    [
        A.Resize(height=int(IMAGE_SIZE), width=int(IMAGE_SIZE), interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ]
)
