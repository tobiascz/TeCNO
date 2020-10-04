from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from albumentations import ImageOnlyTransform
import cv2
import pandas as pd


class Dataset_from_Dataframe_multilabel(Dataset):
    """simple datagenerator from pandas dataframe"""

    # image_path", "class", "time", "video", "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
    def __init__(self, df, transform, label_col, image_path_col="path"):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df[self.image_path_col][index])
        X = np.asarray(X, dtype=np.uint8)
        if self.transform:
            X = self.transform(image=X, mask=None)["image"]
        label = torch.tensor(int(self.df[self.label_col][index]))
        cols = self.df.loc[index, [
            "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors",
            "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
        ]]
        labels_add = torch.tensor(cols)
        return X, label, labels_add


class Dataset_from_Dataframe(Dataset):
    """simple datagenerator from pandas dataframe"""

    # image_path", "class", "time", "video", "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
    def __init__(self,
                 df,
                 transform,
                 label_col,
                 img_root="",
                 image_path_col="path"):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.img_root = img_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        p = self.img_root / self.df[self.image_path_col][index]
        X = Image.open(p)
        X = np.asarray(X, dtype=np.uint8)
        if self.transform:
            X = self.transform(image=X, mask=None)["image"]
        label = torch.tensor(int(self.df[self.label_col][index]))
        X = X.type(torch.FloatTensor)
        return X, label


class AlbuTransformDataset(Dataset):
    r"""
    Dataset with additional transformation.

    Arguments:
        dataset (Dataset): The whole Dataset
        transform: transormation to be applied to the image
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        X = np.asarray(X, np.uint8)
        X = self.transform(image=X, mask=None)["image"]
        return X, y

    def __len__(self):
        return len(self.dataset)


class TransformDataset(Dataset):
    r"""
    Dataset with additional transformation.

    Arguments:
        dataset (Dataset): The whole Dataset
        transform: transormation to be applied to the image
        target_transform: transformation applied to target (y)
    """
    def __init__(self, dataset, transform, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            y = self.target_transform(y)
        return X, y

    def __len__(self):
        return len(self.dataset)


class GrayToRGB(ImageOnlyTransform):
    """
    Targets:
        image

    Image types:
        uint8, float32
    """
    def __init__(self, p=1.0):
        super(GrayToRGB, self).__init__(p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


class RGBToGray(ImageOnlyTransform):
    """
    Targets:
        image

    Image types:
        uint8, float32
    """
    def __init__(self, p=1.0):
        super(RGBToGray, self).__init__(p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def split_dataframe(df_in, stratify_col, test_size=0.1, val_size=0.1, seed=42):
    """split dataframes into train, val and testing using a stratified shuffle algorithm
    :param df_in: panda dataframe to be split
    :param test_size: percentage of total data to be used for testing
    :param val_size: percentage of total data to be used for validation
    :param stratify_col: pandas column to be used for stratification, e.g. the label column
    :param seed: seed used for random state
    """
    df = {}
    df["train"], df["test"] = train_test_split(
        df_in,
        test_size=test_size,
        shuffle=True,
        stratify=df_in[stratify_col],
        random_state=seed,
    )
    if val_size > 0:
        df["train"], df["val"] = train_test_split(
            df["train"],
            test_size=val_size / (1 - test_size),
            shuffle=True,
            stratify=df["train"][stratify_col],
            random_state=seed,
        )
    for key in df.keys():
        df[key] = df[key].reset_index()
    return df


def get_median_frequency_balancing_weights(classes=None, labels=None):
    class_weights = np.zeros_like(classes, dtype=np.float)
    counts = np.zeros_like(classes)
    for i, cat in enumerate(classes):
        counts[i] = len(labels[labels == cat])
    counts = counts.astype(np.float)
    median_freq = np.median(counts)
    for i, count in enumerate(counts):
        class_weights[i] = median_freq / count
    return class_weights
