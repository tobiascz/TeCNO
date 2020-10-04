import pandas as pd
from torch.utils.data import Dataset
import pprint, pickle
from pathlib import Path
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
from utils.dataset_utils import Dataset_from_Dataframe
import torch

class Cholec80FeatureExtract:
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_mode = hparams.dataset_mode
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        self.fps_sampling = hparams.fps_sampling
        self.fps_sampling_test = hparams.fps_sampling_test
        self.cholec_root_dir = Path(self.hparams.data_root +
                                    "/cholec80")  # videos splitted in images
        self.transformations = self.__get_transformations()
        self.class_labels = [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningCoagulation",
            "GallbladderRetraction",
        ]
        weights = [
            1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,
        ]
        self.class_weights = np.asarray(weights)
        self.label_col = "class"
        self.df = {}
        self.df["all"] = pd.read_pickle(
            self.cholec_root_dir / "dataframes/cholec_split_250px_25fps.pkl")

        #print("Drop nan rows from df manually")
        ## Manually remove these indices as they are nan in the DF which causes issues
        index_nan = [1983913, 900090]
        #self.df["all"][self.df["all"].isna().any(axis=1)]
        self.df["all"] = self.df["all"].drop(index_nan)
        assert self.df["all"].isnull().sum().sum(
        ) == 0, "Dataframe contains nan Elements"
        self.df["all"] = self.df["all"].reset_index()

        self.vids_for_training = [i for i in range(1, 41)]
        self.vids_for_val = [i for i in range(41, 49)]
        self.vids_for_test = [i for i in range(49, 81)]

        self.df["train"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_training)]
        self.df["val"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_val)]
        if hparams.test_extract:
            print(
                f"test extract enabled. Test will be used to extract the videos (testset = all)"
            )
            self.vids_for_test = [i for i in range(1, 81)]
            self.df["test"] = self.df["all"]
        else:
            self.df["test"] = self.df["all"][self.df["all"]["video_idx"].isin(
                self.vids_for_test)]

        len_org = {
            "train": len(self.df["train"]),
            "val": len(self.df["val"]),
            "test": len(self.df["test"])
        }
        if self.fps_sampling < 25 and self.fps_sampling > 0:
            factor = int(25 / self.fps_sampling)
            print(
                f"Subsampling data: 25fps --> {self.fps_sampling}fps (factor: {factor})"
            )
            self.df["train"] = self.df["train"].iloc[::factor]
            self.df["val"] = self.df["val"].iloc[::factor]
            self.df["all"] = self.df["all"].iloc[::factor]
            for split in ["train", "val"]:
                print(
                    f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")
        if hparams.fps_sampling_test < 25 and self.fps_sampling_test > 0:
            factor = int(25 / self.fps_sampling_test)
            print(
                f"Subsampling data: 25fps --> {self.fps_sampling}fps (factor: {factor})"
            )
            self.df["test"] = self.df["test"].iloc[::factor]
            split = "test"
            print(f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")

        self.data = {}
        if self.dataset_mode == "img":
            for split in ["train", "val", "test"]:
                self.df[split] = self.df[split].reset_index()
                self.data[split] = Dataset_from_Dataframe(
                    self.df[split],
                    self.transformations[split],
                    self.label_col,
                    img_root=self.cholec_root_dir / "output_split_all",
                    image_path_col="image_path",
                )

        if self.dataset_mode == "img_multilabel":
            for split in ["train", "val"]:
                self.df[split] = self.df[split].reset_index()
                self.data[split] = Dataset_from_Dataframe_multilabel(
                    self.df[split],
                    self.transformations[split],
                    self.label_col,
                    img_root=self.cholec_root_dir / "cholec_split_250px_25fps",
                    image_path_col="image_path",
                    add_label_cols=[
                        "tool_Grasper", "tool_Bipolar", "tool_Hook",
                        "tool_Scissors", "tool_Clipper", "tool_Irrigator",
                        "tool_SpecimenBag"
                    ])
            # here we want to extract all features
            #self.df["test"] = self.df["all"].reset_index()
            self.df["test"] = self.df["test"].reset_index()
            self.data["test"] = Dataset_from_Dataframe_multilabel(
                self.df["test"],
                self.transformations["test"],
                self.label_col,
                img_root=self.cholec_root_dir / "cholec_split_250px_25fps",
                image_path_col="image_path",
                add_label_cols=[
                    "video_idx", "image_path", "index", "tool_Grasper",
                    "tool_Bipolar", "tool_Hook", "tool_Scissors",
                    "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
                ])

        if self.dataset_mode == "vid_multilabel":
            for split in ["train", "val", "test"]:
                self.df[split] = self.df[split].reset_index()
                self.data[split] = Dataset_from_Dataframe_video_based(
                    self.df[split],
                    self.transformations[split],
                    self.label_col,
                    img_root=self.cholec_root_dir / "cholec_split_250px_25fps",
                    image_path_col="image_path",
                )

    def __get_transformations(self):
        norm_mean = [0.3456, 0.2281, 0.2233]
        norm_std = [0.2528, 0.2135, 0.2104]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        training_augmentation = Compose([
            ShiftScaleRotate(shift_limit=0.1,
                             scale_limit=(-0.2, 0.5),
                             rotate_limit=15,
                             border_mode=0,
                             value=0,
                             p=0.7),
        ])

        data_transformations = {}
        data_transformations["train"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            training_augmentation,
            normalize,
            ToTensorV2(),
        ])
        data_transformations["val"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            normalize,
            ToTensorV2(),
        ])
        data_transformations["test"] = data_transformations["val"]
        return data_transformations

    def median_frequency_weights(
            self, file_list):  ## do only once and define weights in class
        frequency = [0, 0, 0, 0, 0, 0, 0]
        for i in file_list:
            frequency[int(i[1])] += 1
        median = np.median(frequency)
        weights = [median / j for j in frequency]
        return weights

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cholec80_specific_args = parser.add_argument_group(
            title='cholec80 specific args options')
        cholec80_specific_args.add_argument("--fps_sampling",
                                            type=float,
                                            default=25)
        cholec80_specific_args.add_argument("--fps_sampling_test",
                                            type=float,
                                            default=25)
        cholec80_specific_args.add_argument(
            "--dataset_mode",
            default='video',
            choices=[
                'vid_multilabel', 'img', 'img_multilabel',
                'img_multilabel_feature_extract'
            ])
        cholec80_specific_args.add_argument("--test_extract",
                                            action="store_true")
        return parser


class Dataset_from_Dataframe_video_based(Dataset):
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
        self.starting_idx = self.df["video_idx"].min()
        print(self.starting_idx)

    def __len__(self):
        return len(self.df.video_idx.unique())

    def __getitem__(self, index):
        sindex = self.starting_idx + index
        img_list = self.df.loc[self.df["video_idx"] == sindex]
        videos_x = torch.zeros([len(img_list), 3, 224, 224], dtype=torch.float)
        label = torch.tensor(img_list[self.label_col].tolist(),
                             dtype=torch.int)
        f_video = self.load_cholec_video(img_list)
        if self.transform:
            for i in range(len(f_video)):
                videos_x[i] = self.transform(image=f_video[i],
                                             mask=None)["image"]
        add_label_cols = [
            "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors",
            "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
        ]
        add_label = []
        for add_l in add_label_cols:
            add_label.append(img_list[add_l].tolist())
        #print(f"Index of video: {index} - sindex: {sindex} - len_label: {label.shape[0]} - len_vid: {videos_x.shape[0]}")
        assert videos_x.shape[0] == label.shape[0], f"weird shapes at {sindex}"
        assert videos_x.shape[
            0] > 0, f"no video returned shape: {videos_x.shape[0]}"
        return videos_x, label, add_label

    def load_cholec_video(self, img_list):
        f_video = []
        allImage = img_list[self.image_path_col].tolist()
        for i in range(img_list.shape[0]):
            p = self.img_root / allImage[i]
            im = Image.open(p)
            f_video.append(np.asarray(im, dtype=np.uint8))
        f_video = np.asarray(f_video)
        return f_video


class Dataset_from_Dataframe_multilabel(Dataset):
    def __init__(self,
                 df,
                 transform,
                 label_col,
                 img_root="",
                 image_path_col="path",
                 add_label_cols=[]):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.img_root = img_root
        self.add_label_cols = add_label_cols
        self.failure_count = 3

    def __len__(self):
        return len(self.df)

    def load_from_path(self, index):
        img_path_df = self.df.loc[index, self.image_path_col]
        p = self.img_root / img_path_df
        X = Image.open(p)
        X_array = np.array(X)
        return X_array, p

    def __getitem__(self, index):
        for i in range(self.failure_count):
            X_array, p = self.load_from_path(index + i)
            test_shape = ()
            if X_array.shape != (250, 250, 3):
                print(f"\nerror at index: {index}")
                print(f"Path to img: {p}")
                print(f"x_array shape: {X_array.shape}")
                print(f"try: {i}/{self.failure_count}")
                print(f"Exists: {p.exists()}")
                continue
            if self.transform:
                X = self.transform(image=X_array, mask=None)["image"]
            label = torch.tensor(int(self.df[self.label_col][index]))
            add_label = []
            for add_l in self.add_label_cols:
                add_label.append(self.df[add_l][index])
            X = X.type(torch.FloatTensor)
            return X, label, add_label
        print("Too much loading failed ... Quitting")
        raise ArithmeticError
