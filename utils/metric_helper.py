import torch
from pycm import ConfusionMatrix
import numpy as np
from typing import Any, Callable, Optional, Union

from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.utils import _input_format_classification

from pytorch_lightning.metrics import Precision, Recall


'''The two following measures are defined per phase: Recall is defined as the number of correct detections inside the 
ground truth phase divided by its length. Precision is the sum of correct detec- tions divided by the number of 
correct and incorrect detections. This is complementary to recall by indicating whether parts of other phases are 
detected incorrectly as the considered phase. To present summarized results, we will use accuracy together with 
average recall and average precision, corresponding to recall and precision averaged over all phases. Since the phase 
lengths can vary largely, incorrect detections inside short phases tend to be hidden within the accuracy, 
but are revealed within precision and recall'''





class PrecisionOverClasses(Precision):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, average: str = 'micro', multilabel: bool = False,
            compute_on_step: bool = True, dist_sync_on_step: bool = False, process_group: Optional[Any] = None, ):
        super().__init__(num_classes=num_classes, threshold=threshold, average=average, multilabel=multilabel,
                         compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group)
    def compute(self):
        return self.true_positives.float() / self.predicted_positives

class RecallOverClasse(Recall):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, average: str = 'micro', multilabel: bool = False,
            compute_on_step: bool = True, dist_sync_on_step: bool = False, process_group: Optional[Any] = None, ):
        super().__init__(num_classes=num_classes, threshold=threshold, average=average, multilabel=multilabel,
                         compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group)
    def compute(self):
        return self.true_positives.float() / self.actual_positives


class AccuracyStages(Metric):
    def __init__(self, num_stages=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_stages = num_stages
        for s in range(self.num_stages):
            self.add_state(f"S{s + 1}_correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.total += target.numel()
        for s in range(self.num_stages):
            preds_stage, target = _input_format_classification(preds[s], target, threshold=0.5)
            assert preds_stage.shape == target.shape

            s_correct = getattr(self, f"S{s + 1}_correct")
            s_correct += torch.sum(preds_stage == target)
            setattr(self, f"S{s + 1}_correct", s_correct)

    def compute(self):
        acc_list = []
        for s in range(self.num_stages):
            s_correct = getattr(self, f"S{s + 1}_correct")
            acc_list.append(s_correct.float() / self.total)
        return acc_list


def calc_average_over_metric(metric_list, normlist):
    for i in metric_list:
        metric_list[i] = np.asarray([0 if value == "None" else value for value in metric_list[i]])
        if normlist[i] == 0:
            metric_list[i] = 0  # TODO: correct?
        else:
            metric_list[i] = metric_list[i].sum() / normlist[i]
    return metric_list


def create_print_output(print_dict, space_desc, space_item):
    msg = ""
    for key, value in print_dict.items():
        msg += f"{key:<{space_desc}}"
        for i in value:
            msg += f"{i:>{space_item}}"
        msg += "\n"
    msg = msg[:-1]
    return msg


