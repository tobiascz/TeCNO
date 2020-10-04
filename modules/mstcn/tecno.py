import logging
from collections import OrderedDict
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from utils.metric_helper import temporal_cholec80_metric
from losses.classification import classification_loss


class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(TeCNO, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.weights_train = self.dataset.weights["train"]
        self.max_metric = [
            0.0, "no_max_last_epoch_yet", 0.0, "no_total_max_yet"
        ]

    def forward(self, x):
        video_fe = x.transpose(2, 1)
        y_classes = self.model.forward(video_fe)
        y_classes = torch.softmax(y_classes, dim=2)
        return y_classes

    def loss_function(self, labels, y_classes):
        loss = classification_loss(self, labels, y_classes)
        return loss

    def get_class_acc(self, y_true, y_classes):
        y_true = y_true.squeeze()
        y_classes = y_classes.squeeze()
        y_classes = torch.argmax(y_classes, dim=0)
        acc_classes = torch.sum(
            y_true == y_classes).float() / (y_true.shape[0] * 1.0)
        return acc_classes

    def get_class_acc_each_layer(self, y_true, y_classes):
        y_true = y_true.squeeze()
        accs_classes = []
        for i in range(y_classes.shape[0]):
            acc_classes = self.get_class_acc(y_true, y_classes[i, 0])
            accs_classes.append(acc_classes)
        return accs_classes

    def training_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_classes = self.forward(stem)
        train_loss = self.loss_function(y_true, y_classes)
        train_acc_classes = self.get_class_acc(y_true, y_classes[-1])
        tqdm_dict = {
            "train_loss": train_loss,
            "train_acc_c": train_acc_classes
        }
        output = OrderedDict({
            "loss": train_loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "train_acc_c": train_acc_classes,
        })
        return output

    def validation_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_classes = self.forward(stem)

        loss = self.loss_function(y_true, y_classes)
        val_acc_classes = self.get_class_acc(y_true, y_classes[-1])
        val_accs_classes = self.get_class_acc_each_layer(y_true, y_classes)
        output = OrderedDict({
            "val_loss": loss,
            "val_acc_c": val_acc_classes,
            "val_accs_c": val_accs_classes,
            "y_true": y_true,
            "y_classes": y_classes,
        })
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        output_metric = temporal_cholec80_metric(outputs)
        if output_metric['max_acc_total'] > self.max_metric[2]:
            self.max_metric[2] = output_metric["max_acc_total"]
            self.max_metric[3] = self.current_epoch

        if output_metric["max_acc_last_stage"] > self.max_metric[0]:
            print(
                f"new max in last stage: {output_metric['max_acc_last_stage']} at Epoch: {self.current_epoch}"
            )
            self.max_metric[0] = output_metric["max_acc_last_stage"]
            self.max_metric[1] = self.current_epoch
        print(
            f"epoch: {self.current_epoch} | current max last stage: {self.max_metric[0]:.4f} | total max:"
            f" {self.max_metric[2]:.4f} - e:{self.max_metric[3]}")

        mean_metric_keys = ["val_loss", "val_acc_c"]
        means = {key: 0 for key in mean_metric_keys}
        for output in outputs:
            for key in mean_metric_keys:
                means[key] += output[key]
        means = {
            key: (means[key] / (len(outputs) * 1.0))
            for key in mean_metric_keys
        }

        mean_metric2 = ["val_accs_c"]
        means2 = {
            key: torch.zeros(len(output["val_accs_c"]), dtype=torch.float)
            for key in mean_metric2
        }
        for key in mean_metric2:
            for i, output in enumerate(outputs):
                to_add = torch.FloatTensor(output[key])
                means2[key] += to_add
        means2 = {
            key: (means2[key] / (len(outputs) * 1.0))
            for key in mean_metric2
        }
        if abs(means2["val_accs_c"][-1] - means["val_acc_c"]) > 0.1:
            print(f"Difference tooo big something is STRANGE!")
        tqdm_dict = {**means}

        log_dict = tqdm_dict.copy()
        for i in range(len(output["val_accs_c"])):
            log_dict[f"val_accs_c{i}"] = means2["val_accs_c"][i]
        log_dict["val_precision_c"] = output_metric["PPV_c"]
        log_dict["val_acc_max_last"] = self.max_metric[0]
        log_dict["val_acc_max_total"] = self.max_metric[2]

        result = {
            "progress_bar": tqdm_dict,
            "log": log_dict,
            "val_loss": means["val_loss"],
            "val_acc_c": means["val_acc_c"],
        }

        return result

    def test_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_classes = self.forward(stem)

        loss = self.loss_function(y_true, y_classes)
        test_acc_classes = self.get_class_acc(y_true, y_classes[-1])

        test_accs_classes = self.get_class_acc_each_layer(y_true, y_classes)
        # print(f"validation step: {val_accs_classes} - {val_accs_regression}")

        output = OrderedDict({
            "test_loss": loss,
            "test_acc_c": test_acc_classes,
            "test_accs_c": test_accs_classes,
            "y_true": y_true,
            "y_classes": y_classes,
        })
        return output

    def test_epoch_end(self, outputs):
        output_metric = temporal_cholec80_metric(outputs)
        mean_metric_keys = ["test_loss", "test_acc_c"]
        means = {key: 0 for key in mean_metric_keys}
        for output in outputs:
            for key in mean_metric_keys:
                means[key] += output[key]
        means = {
            key: (means[key] / (len(outputs) * 1.0))
            for key in mean_metric_keys
        }

        mean_metric2 = ["test_accs_c"]
        means2 = {
            key: torch.zeros(len(output["test_accs_c"]), dtype=torch.float)
            for key in mean_metric2
        }
        for key in mean_metric2:
            for i, output in enumerate(outputs):
                # to_add = np.asarray(output[key])
                to_add = torch.FloatTensor(output[key])
                # print(f"to_add: {to_add}")
                means2[key] += to_add
        means2 = {
            key: (means2[key] / (len(outputs) * 1.0))
            for key in mean_metric2
        }

        tqdm_dict = {**means}

        log_dict = tqdm_dict.copy()
        for i in range(len(output["test_accs_c"])):
            log_dict[f"test_accs_c{i}"] = means2["test_accs_c"][i].clone(
            ).detach()
        log_dict["test_recall_c"] = output_metric["TPR_c"]
        log_dict["test_precision_c"] = output_metric["PPV_c"]
        log_dict["test_acc_max"] = log_dict[f"test_accs_c{i}"]

        result = {
            "progress_bar": tqdm_dict,
            "log": log_dict,
            "test_loss": means["test_loss"],
            "test_acc_c": means["test_acc_c"],
        }

        out_print = f"TEST RESULTS: "
        for key in ["test_acc_max", "test_recall_c", "test_precision_c"]:
            out_print += f"{key}: {float(log_dict[key]):.4f} | "
        print(out_print)

        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = False
        if split == "train":
            should_shuffle = True
        print(f"split: {split} - shuffle: {should_shuffle}")
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    @pl.data_loader
    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    @pl.data_loader
    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        regressiontcn = parser.add_argument_group(
            title='regression tcn specific args options')
        regressiontcn.add_argument("--learning_rate",
                                   default=0.001,
                                   type=float)
        regressiontcn.add_argument("--optimizer_name",
                                   default="adam",
                                   type=str)
        regressiontcn.add_argument("--batch_size", default=1, type=int)

        return parser
