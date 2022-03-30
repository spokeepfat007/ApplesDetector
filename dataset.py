"""
Creates a Pytorch dataset to load own data
"""

import torch
import pandas as pd
from PIL import Image
import numpy as np
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
            self, num_of_classes, csv_file, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.num_of_classes = num_of_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path, label = self.annotations.iloc[index]
        image = Image.open(img_path)
        image = np.array(image.convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label


def create_csv():
    df = pd.DataFrame({'image_path': [], 'label': []})
    data_path = "MyData"
    for folder in os.listdir(data_path):
        for filename in os.listdir(data_path + "/" + folder):
            df2 = {'image_path': data_path + "/" + folder + "/" + filename, 'label': folder}
            df = df.append(df2, ignore_index=True)
    df = df.sample(frac=1)
    print(df.head(10))
    train = df[:int(0.7*len(df))]
    test = df[int(0.7*len(df)):]
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    create_csv()
