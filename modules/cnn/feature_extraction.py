import logging
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pathlib import Path
from torch import nn
from pycm import ConfusionMatrix
import numpy as np
import pickle


class FeatureExtraction(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(FeatureExtraction, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        # get median frequency weights from dataset and put on GPU if required
        self.class_weights = self.dataset.class_weights
        if self.dataset.class_weights is not None and self.hparams.gpus != 0:
            self.class_weights = torch.from_numpy(
                self.class_weights).float().cuda()

        self.num_tasks = self.hparams.num_tasks  # output stem 0, output phase 1 , output phase and tool 2
        self.log_vars = nn.Parameter(torch.zeros(2))

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.sig_f = nn.Sigmoid().cuda()
        self.current_video_idx = self.dataset.df["test"].video_idx.min()
        print(f"starting video idx for testing: {self.current_video_idx}")

        # store model
        self.current_stems = []
        self.current_phase_labels = []
        self.current_p_phases = []
        self.len_test_data = len(self.dataset.data["test"])
        self.model = model
        self.best_metrics_high = {"val_acc_phase": 0}
        self.test_acc_per_video = {}
        self.pickle_path = None

    def set_export_pickle_path(self):
        self.pickle_path = self.hparams.output_path / "cholec80_pickle_export"
        self.pickle_path.mkdir(exist_ok=True)
        print(f"setting export_pickle_path: {self.pickle_path}")

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        if self.num_tasks == 1:
            return self.forward_phase(x)

        if self.num_tasks == 2:
            return self.forward_phase_tool(x)

    def forward_phase(self, x):
        _, phase, _ = self.model.forward(x)
        return phase

    def forward_phase_tool(self, x):
        _, phase, tool = self.model.forward(x)
        return phase, tool

    def forward_stem(self, x):
        stem, phase, _ = self.model.forward(x)
        return stem, phase

    def loss_phase(self, labels, logits):
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        return loss

    def loss_phase_tool(self, labels_phase, labels_tool, p_phase, p_tool):
        loss = 0
        loss_phase = self.loss_phase(labels_phase, p_phase)

        labels_tool = torch.stack(labels_tool, dim=1)
        loss_tools = self.bce_loss(p_tool, labels_tool.data.float())
        # automatic balancing
        precision1 = torch.exp(-self.log_vars[0])
        loss_phase_l = precision1 * loss_phase + self.log_vars[0]
        precision2 = torch.exp(-self.log_vars[1])
        loss_tool_l = precision2 * loss_tools + self.log_vars[1]
        loss = loss_phase_l + loss_tool_l
        return loss

    def get_metrics(self,
                    y_phase=None,
                    y_tool=None,
                    p_phase=None,
                    p_tool=None,
                    val=False):
        metrics = {}
        prediction_phase = torch.argmax(p_phase, dim=1)
        val_str = ""
        if val:
            val_str = "val_"
        if torch.unique(prediction_phase)[0] == torch.unique(
                y_phase)[0] and len(torch.unique(y_phase)) == 1 and len(
                    torch.unique(prediction_phase)) == 1:
            acc_phase = 1.0
            f1_phase = 1.0
        else:
            cm_phase = ConfusionMatrix(
                actual_vector=y_phase.cpu().numpy(),
                predict_vector=prediction_phase.cpu().numpy(),
            )
            acc_phase = cm_phase.Overall_ACC
            f1_phase = cm_phase.F1_Micro

        if type(y_tool) != type(None) and type(p_tool) != type(None):
            sig_out = self.sig_f(p_tool)
            preds_1 = sig_out.cpu() > 0.5
            preds_1 = preds_1.long()
            y_tool = torch.stack(y_tool, dim=1)
            acc_tool = (torch.sum(preds_1 == y_tool.cpu().long()).float() /
                        (preds_1.shape[0] * 1.0)) / 7.0
            metrics[f"{val_str}acc_tool"] = acc_tool
        metrics[f"{val_str}acc_phase"] = torch.tensor(acc_phase)
        metrics[f"{val_str}f1_phase"] = torch.tensor(f1_phase)
        return metrics

    def prediction_phase(self, x, y_phase, val=False):
        p_phase = self.forward_phase(x)
        loss = self.loss_phase(y_phase, p_phase)
        metrics = self.get_metrics(y_phase=y_phase, p_phase=p_phase, val=val)
        #metrics["loss"] = loss
        return loss, metrics

    def prediction_phase_tool(self, x, y_phase, y_tool, val=False):
        p_phase, p_tool = self.forward_phase_tool(x)
        loss = self.loss_phase_tool(y_phase, y_tool, p_phase, p_tool)
        metrics = self.get_metrics(y_phase=y_phase,
                                   y_tool=y_tool,
                                   p_phase=p_phase,
                                   p_tool=p_tool,
                                   val=val)
        #metrics["loss"] = loss
        return loss, metrics

    def training_step(self, batch, batch_idx):
        x, y_phase, y_tool = batch
        if self.num_tasks == 1:
            loss, metrics = self.prediction_phase(x, y_phase)
        else:
            loss, metrics = self.prediction_phase_tool(x, y_phase, y_tool)
        log = metrics
        log.update({"train_loss": loss})
        output = OrderedDict({"loss": loss, "progress_bar": log, "log": log})
        return output

    def get_mean_metrics(self, outputs, val=False):
        mean_metric_keys = outputs[0]["progress_bar"].keys()
        means = {key: 0 for key in mean_metric_keys}
        for output in outputs:
            for key in mean_metric_keys:
                means[key] += output["progress_bar"][key]
        means = {key: means[key] / len(outputs) for key in mean_metric_keys}
        if val:
            loss_mean = np.asarray(
                [k['val_loss'].cpu().numpy() for k in outputs]).mean()
        else:
            loss_mean = np.asarray([k['loss'].cpu().numpy()
                                    for k in outputs]).mean()
        return means, loss_mean

    def training_epoch_end(self, outputs):
        logs, train_loss_mean = self.get_mean_metrics(outputs)
        logs.update({"train_loss_mean": train_loss_mean})
        results = {
            'train_loss': train_loss_mean,
            'progress_bar': logs,
            'log': logs
        }
        return results

    def validation_step(self, batch, batch_idx):
        x, y_phase, y_tool = batch
        if self.num_tasks == 1:
            val_loss, log = self.prediction_phase(x, y_phase, val=True)
        else:
            val_loss, log = self.prediction_phase_tool(x,
                                                       y_phase,
                                                       y_tool,
                                                       val=True)
        log.update({"val_loss": val_loss})
        output = OrderedDict({
            "val_loss": val_loss,
            "progress_bar": log,
            "log": log
        })
        return output

    def validation_epoch_end(self, outputs):
        logs, val_loss_mean = self.get_mean_metrics(outputs, val=True)
        logs.update({"val_loss_mean": val_loss_mean})
        result = {"val_loss": val_loss_mean, "progress_bar": logs, "log": logs}

        for k, v in self.best_metrics_high.items():
            if logs[k] > v:
                self.best_metrics_high[k] = logs[k]
                logs.update({k + "_best": logs[k]})

        if self.hparams.on_polyaxon:
            out_print = f"Epoch: {self.current_epoch} | "
            for key in logs:
                out_print += f"{key}: {float(logs[key]):.2f} | "
            print(out_print)
        return result

    def get_phase_acc(self, true_label, pred, during_train=False):
        pred = torch.FloatTensor(pred)
        pred_phase = torch.softmax(pred, dim=1)
        labels_pred = torch.argmax(pred_phase, dim=1)
        true_label = torch.IntTensor(true_label)
        acc_phase = torch.sum(
            true_label == labels_pred.int()).float() / (len(true_label) * 1.0)
        if not during_train:
            cm = ConfusionMatrix(
                actual_vector=true_label.cpu().numpy(),
                predict_vector=labels_pred.cpu().numpy(),
            )
            f1 = cm.F1_Macro
            ppv = cm.PPV
            tpr = cm.TPR
            keys = cm.classes
            return float(acc_phase.cpu().numpy()), ppv, tpr, keys, f1
        return float(acc_phase.cpu().numpy()), 0, 0, 0, 0

    def save_to_drive(self, vid_index):
        acc, ppv, tpr, keys, f1 = self.get_phase_acc(self.current_phase_labels,
                                                     self.current_p_phases)
        save_path = self.pickle_path / f"{self.hparams.fps_sampling_test}fps"
        save_path.mkdir(exist_ok=True)
        save_path_txt = save_path / f"video_{vid_index}_{self.hparams.fps_sampling_test}fps_acc.txt"
        save_path_vid = save_path / f"video_{vid_index}_{self.hparams.fps_sampling_test}fps.pkl"

        with open(save_path_txt, "w") as f:
            f.write(
                f"vid: {vid_index}; acc: {acc}; ppv: {ppv}; tpr: {tpr}; keys: {keys}; f1: {f1}"
            )
            self.test_acc_per_video[vid_index] = acc
            print(
                f"save video {vid_index} | acc: {acc:.4f}; ppv: {ppv}; tpr: {tpr}; keys: {keys}; f1: {f1}"
            )
        with open(save_path_vid, 'wb') as f:
            pickle.dump([
                np.asarray(self.current_stems),
                np.asarray(self.current_p_phases),
                np.asarray(self.current_phase_labels)
            ], f)

    def test_step(self, batch, batch_idx):

        x, y, (vid_idx, img_name, img_index, tool_Grasper, tool_Bipolar,
               tool_Hook, tool_Scissors, tool_Clipper, tool_Irrigator,
               tool_SpecimenBag) = batch
        y = y.cpu().numpy()
        vid_idx_raw = vid_idx.cpu().numpy()

        with torch.no_grad():
            stem, y_hat = self.forward_stem(x)

        y_hat = np.asarray(y_hat.cpu())
        y_hat = y_hat.squeeze()

        vid_idxs, indexes = np.unique(vid_idx_raw, return_index=True)
        vid_idxs = [int(x) for x in vid_idxs]
        index_next = len(vid_idx) if len(vid_idxs) == 1 else indexes[1]
        for i in range(len(vid_idxs)):
            vid_idx = vid_idxs[i]
            index = indexes[i]
            if vid_idx != self.current_video_idx:
                self.save_to_drive(self.current_video_idx)
                self.current_stems = []
                self.current_phase_labels = []
                self.current_p_phases = []
                if len(vid_idxs) <= i + 1:
                    index_next = len(vid_idx_raw)
                else:
                    index_next = indexes[i+1]  # for the unlikely case that we have 3 phases in one batch
                self.current_video_idx = vid_idx
            self.current_p_phases.extend(
                np.asarray(y_hat[index:index_next, :]).tolist())
            self.current_stems.extend(
                stem[index:index_next, :].cpu().detach().numpy().tolist())
            self.current_phase_labels.extend(
                np.asarray(y[index:index_next]).tolist())

        if (batch_idx + 1) * self.hparams.batch_size >= self.len_test_data:
            self.save_to_drive(vid_idx)
            print(f"Finished extracting all videos...")

        batch_acc, _, _, _, _ = self.get_phase_acc(y, y_hat, during_train=True)
        tensorboard_logs = {'batch_idx': batch_idx}
        return {'test_acc': batch_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        mean_metric_keys = ["test_acc"]
        means = {key: 0 for key in mean_metric_keys}
        for output in outputs:
            for key in mean_metric_keys:
                means[key] += output[key]
        means = {key: means[key] / len(outputs) for key in mean_metric_keys}


        means["test_acc_train"] = np.mean(
            np.asarray([
                self.test_acc_per_video[x]
                for x in self.dataset.vids_for_training
            ]))
        means["test_acc_val"] = np.mean(
            np.asarray([
                self.test_acc_per_video[x]
                for x in self.dataset.vids_for_val
            ]))
        means["test_acc_test"] = np.mean(
            np.asarray([
                self.test_acc_per_video[x]
                for x in self.dataset.vids_for_test
            ]))


        print(f"Done Extracting the overall acc is: {means['test_acc']:.2f}")

        result = {
            "test_acc_phase_all": means["test_acc"],
            "test_acc_vid_testset": means["test_acc_test"],
            "test_acc_vid_valset": means["test_acc_val"],
            "test_acc_vid_trainset": means["test_acc_train"],
        }

        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None

        batch_size_final = self.hparams.batch_size
        if self.hparams.batch_size > self.hparams.model_specific_batch_size_max:
            print(
                f"The choosen batchsize is too large for this model."
                f" It got automatically reduced from: {self.hparams.batch_size} to {self.hparams.model_specific_batch_size_max}"
            )
            self.hparams.batch_size = self.hparams.model_specific_batch_size_max

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        if split == "val" or split == "test":
            should_shuffle = False
        print(f"split: {split} - shuffle: {should_shuffle}")
        worker = self.hparams.num_workers
        if split == "test":
            print(
                "worker set to 0 due to test"
            )  # otherwise for extraction the order in which data is loaded is not sorted e.g. 1,2,3,4, --> 1,5,3,2
            worker = 0

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=worker,
            pin_memory=True,
        )
        return loader

    @pl.data_loader
    def train_dataloader(self):
        """
        Intialize train dataloader
        :return: train loader
        """
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        """
        Initialize val loader
        :return: validation loader
        """
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    @pl.data_loader
    def test_dataloader(self):
        """
        Initialize test loader
        :return: test loader
        """
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        self.set_export_pickle_path()
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        cholec_fe_module = parser.add_argument_group(
            title='cholec_fe_module specific args options')
        cholec_fe_module.add_argument("--learning_rate",
                                      default=0.001,
                                      type=float)
        cholec_fe_module.add_argument("--num_tasks",
                                      default=1,
                                      type=int,
                                      choices=[1, 2])
        cholec_fe_module.add_argument("--optimizer_name",
                                      default="adam",
                                      type=str)
        cholec_fe_module.add_argument("--batch_size", default=32, type=int)
        return parser
