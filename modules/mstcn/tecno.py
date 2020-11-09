import logging
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pytorch_lightning.metrics.utils import _input_format_classification
from pytorch_lightning.core.lightning import LightningModule
from utils.metric_helper import AccuracyStages, RecallOverClasse, PrecisionOverClasses
from torch import nn
import numpy as np


class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(TeCNO, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.weights_train = np.asarray(self.dataset.weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float())
        self.init_metrics()

    def init_metrics(self):
        self.train_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.val_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0 , "stage": 0, "last_stage_max_acc_is_global": False}

        self.precision_metric = PrecisionOverClasses(num_classes=7)
        self.recall_metric = RecallOverClasse(num_classes=7)
        #self.cm_metric = ConfusionMatrix(num_classes=10)

    def forward(self, x):
        video_fe = x.transpose(2, 1)
        y_classes = self.model.forward(video_fe)
        y_classes = torch.softmax(y_classes, dim=2)
        return y_classes

    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):  ### make the interuption free stronge the more layers.
            p_classes = y_classes[j].squeeze().transpose(1, 0)
            ce_loss = self.ce_loss(p_classes, labels.squeeze())
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)
        return clc_loss

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


    '''def log_precision_and_recall(self, precision, recall, step):
        for n,p in enumerate(precision):
            if not p.isnan():
                self.log(f"{step}_precision_{self.dataset.class_labels[n]}",p ,on_step=True, on_epoch=True)
        for n,p in enumerate(recall):
            if not p.isnan():
                self.log(f"{step}_recall_{self.dataset.class_labels[n]}",p ,on_step=True, on_epoch=True)'''

    def calc_precision_and_recall(self, y_pred, y_true, step="val"):
        y_max_pred, y_true = _input_format_classification(y_pred[-1], y_true, threshold=0.5)
        precision = self.precision_metric(y_max_pred, y_true)
        recall = self.recall_metric(y_max_pred, y_true)
        #if step == "val":
        #    self.log_precision_and_recall(precision, recall, step=step)
        return precision, recall

    def log_average_precision_recall(self, outputs, step="val"):
        precision_list = [o["precision"] for o in outputs]
        recall_list = [o["recall"] for o in outputs]
        x = torch.stack(precision_list)
        y = torch.stack(recall_list)
        phase_avg_precision = [torch.mean(x[~x[:, n].isnan(), n]) for n in range(x.shape[1])]
        phase_avg_recall = [torch.mean(y[~y[:, n].isnan(), n]) for n in range(x.shape[1])]
        phase_avg_precision = torch.stack(phase_avg_precision)
        phase_avg_recall = torch.stack(phase_avg_recall)
        phase_avg_precision_over_video = phase_avg_precision[~phase_avg_precision.isnan()].mean()
        phase_avg_recall_over_video = phase_avg_recall[~phase_avg_recall.isnan()].mean()
        self.log(f"{step}_avg_precision", phase_avg_precision_over_video, on_epoch=True, on_step=False)
        self.log(f"{step}_avg_recall", phase_avg_recall_over_video, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)
        loss = self.loss_function(y_pred, y_true)
        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="train")
        acc_stages=self.train_acc_stages(y_pred, y_true)
        acc_stages_dict = {f"train_S{s+1}_acc":acc_stages[s] for s in range(len(acc_stages))}
        acc_stages_dict["train_acc"] = acc_stages_dict.pop(f"train_S{len(acc_stages)}_acc") # Renaming metric of last Stage
        self.log_dict(acc_stages_dict, on_epoch=True, on_step=False)
        return {"loss":loss, "precision": precision, "recall": recall}


    def training_epoch_end(self, outputs):
        self.log_average_precision_recall(outputs, step="train")


    def validation_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="val")
        self.val_acc_stages(y_pred, y_true)
        acc_stages = self.val_acc_stages.compute()
        metric_dict = {f"val_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        metric_dict["val_acc"] = metric_dict.pop(f"val_S{len(acc_stages)}_acc") # Renaming metric of last Stage
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        return metric_dict


    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0 , "stage": 0}
        """
        val_acc_stage_last_epoch = torch.stack([o["val_acc"] for o in outputs]).mean()

        if val_acc_stage_last_epoch > self.max_acc_last_stage["acc"]:
            self.max_acc_last_stage["acc"] = val_acc_stage_last_epoch
            self.max_acc_last_stage["epoch"] = self.current_epoch

        self.log("val: max acc last Stage", self.max_acc_last_stage["acc"])
        self.log_average_precision_recall(outputs, step="val")




    def test_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="test")
        self.val_acc_stages(y_pred, y_true)
        acc_stages = self.val_acc_stages.compute()
        metric_dict = {f"test_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        metric_dict["test_acc"] = metric_dict.pop(f"test_S{len(acc_stages)}_acc") # Renaming metric of last Stage
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        return metric_dict


    def test_epoch_end(self, outputs):
        test_acc = torch.stack([o["test_acc"] for o in outputs]).mean()
        self.log("test_acc", test_acc)
        self.log_average_precision_recall(outputs, step="test")



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = False
        if split == "train":
            should_shuffle = True
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)
            should_shuffle = False
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

    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

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
