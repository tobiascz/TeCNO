from pathlib import Path
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd


class Cholec80Helper(Dataset):
    def __init__(self, hparams, data_p, dataset_split=None):
        assert dataset_split != None
        self.data_p = data_p
        assert hparams.data_root != ""
        self.data_root = Path(hparams.data_root)
        self.number_vids = len(self.data_p)
        self.dataset_split = dataset_split
        self.factor_sampling = hparams.factor_sampling

    def __len__(self):
        return self.number_vids

    def __getitem__(self, index):
        vid_id = index
        p = self.data_root / self.data_p[vid_id]
        unpickled_x = pd.read_pickle(p)
        stem = np.asarray(unpickled_x[0],
                          dtype=np.float32)[::self.factor_sampling]
        y_hat = np.asarray(unpickled_x[1],
                           dtype=np.float32)[::self.factor_sampling]
        y = np.asarray(unpickled_x[2])[::self.factor_sampling]
        return stem, y_hat, y


class Cholec80():
    def __init__(self, hparams):
        self.name = "Cholec80Pickle"
        self.hparams = hparams
        self.class_labels = [
            "Preparation",
            "CalotTriangleDissection",
            "ClippingCutting",
            "GallbladderDissection",
            "GallbladderPackaging",
            "CleaningCoagulation",
            "GallbladderRetraction",
        ]
        self.out_features = self.hparams.out_features
        self.features_per_seconds = hparams.features_per_seconds
        hparams.factor_sampling = (int(25 / hparams.features_subsampling))
        print(
            f"Subsampling features: 25features_ps --> {hparams.features_subsampling}features_ps (factor: {hparams.factor_sampling})"
        )

        self.data_p = {}
        self.data_p["train"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i:02d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(1, 41)]
        self.data_p["val"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i:02d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(41, 49)]
        self.data_p["test"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i:02d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(49, 81)]

        # Datasplit is equal to Endonet and Multi-Task Recurrent ConvNet with correlation loss for surgical vid analysis
        self.weights = {}
        self.weights["train"] = [
            1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,
        ]
        self.weights["train_log"] = [1.25, 0.5, 1.0, 0.25, 1.25, 1., 1.5]

        self.data = {}
        for split in ["train", "val", "test"]:
            self.data[split] = Cholec80Helper(hparams,
                                              self.data_p[split],
                                              dataset_split=split)

        print(
            f"train size: {len(self.data['train'])} - val size: {len(self.data['val'])} - test size:"
            f" {len(self.data['test'])}")

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cholec80_specific_args = parser.add_argument_group(
            title='cholec80 dataset specific args options')
        cholec80_specific_args.add_argument("--features_per_seconds",
                                                  default=25,
                                                  type=float)
        cholec80_specific_args.add_argument("--features_subsampling",
                                                  default=5,
                                                  type=float)

        return parser
