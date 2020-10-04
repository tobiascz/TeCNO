import torch
from pycm import ConfusionMatrix
import numpy as np


def calc_average_over_metric(metric_list, normlist):
    for i in metric_list:
        metric_list[i] = np.asarray(
            [0 if value == "None" else value for value in metric_list[i]])
        if normlist[i] == 0:
            metric_list[i] = 0  #TODO: correct?
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


def temporal_cholec80_metric(outputs, test_mode=False):
    # calc precision: ppv classwise
    # calc recall : tpr classwise

    ppv_list_c = {key: [0] * len(outputs) for key in range(7)}
    tpr_list_c = {key: [0] * len(outputs) for key in range(7)}

    list_c_acc_avg = np.zeros(outputs[-1]["y_classes"].shape[0])
    list_c_f1_avg = np.zeros(outputs[-1]["y_classes"].shape[0])
    norm_list_c = [0, 0, 0, 0, 0, 0, 0]
    norm_list_r = [0, 0, 0, 0, 0, 0, 0]

    for num_cur_out, output in enumerate(outputs):  #
        y_true = output["y_true"].squeeze().cpu().numpy()
        y_classes_output = output["y_classes"]
        list_c_acc = []
        list_c_f1 = []
        keys_c = []
        c_ppv = None
        c_tpr = None

        for i in range(y_classes_output.shape[0]):
            y_classes_maxed = torch.argmax(y_classes_output[i, 0].squeeze(),
                                           dim=0).cpu().numpy()
            if len(np.unique(y_true)) == 1 and len(
                    np.unique(y_classes_maxed)) == 1 and np.all(
                        y_true == y_classes_maxed):
                list_c_acc.append(1.0)
                list_c_f1.append(1.0)
                c_ppv = {np.unique(y_true)[0]: 1.0}
                c_tpr = {np.unique(y_true)[0]: 1.0}
                keys_c = np.unique(y_true)
            else:
                conf_m_c = ConfusionMatrix(actual_vector=y_true,
                                           predict_vector=y_classes_maxed)
                list_c_acc.append(conf_m_c.Overall_ACC)
                list_c_f1.append(conf_m_c.F1_Macro)
                c_ppv = conf_m_c.PPV  # always the last
                c_tpr = conf_m_c.TPR
                keys_c = conf_m_c.classes

        list_c_acc_avg += np.asarray(list_c_acc)
        list_c_f1_avg += np.asarray(list_c_f1)

        for key in keys_c:
            norm_list_c[key] += 1
            ppv_list_c[key][num_cur_out] = c_ppv[key]
            tpr_list_c[key][num_cur_out] = c_tpr[key]

    num_outputs = num_cur_out + 1
    list_c_acc_avg = list_c_acc_avg / num_outputs
    list_c_f1_avg = list_c_f1_avg / num_outputs

    ppv_list_c = calc_average_over_metric(ppv_list_c, norm_list_c)
    tpr_list_c = calc_average_over_metric(tpr_list_c, norm_list_c)

    #### Averaged over all classes now i have to average over all phaases
    ppv_list_c = np.fromiter(ppv_list_c.values(), dtype=float)
    tpr_list_c = np.fromiter(tpr_list_c.values(), dtype=float)
    PPV_c = np.asarray(ppv_list_c).sum() / len(ppv_list_c)
    TPR_c = np.asarray(tpr_list_c).sum() / len(tpr_list_c)

    num_stages = len(list_c_acc)
    space_desc = 15
    space_item = 10
    print(f"\n{'StageMetrics':-^{space_desc+(num_stages*space_item)}}")
    print_dict = {
        "stage": np.arange(1, num_stages + 1),
        "list_acc_c": np.around(list_c_acc_avg, decimals=4),
        "list_f1_c": np.around(list_c_f1_avg, decimals=4)
    }

    string_desc = create_print_output(print_dict, space_desc, space_item)
    print(string_desc)

    print(f"{'OverallMetrics':-^{space_desc+(num_stages*space_item)}}")
    print_dict_overall = {
        "PPV": np.around(np.asarray([PPV_c]), decimals=4),
        "TPR": np.around(np.asarray([TPR_c]), decimals=4)
    }
    string_desc_overall = create_print_output(print_dict_overall, space_desc,
                                              space_item)
    print(string_desc_overall)

    max_acc_total = np.max(list_c_acc_avg)
    max_acc_last_stage = list_c_acc_avg[-1]

    output = {
        "PPV_c": PPV_c,
        "TPR_c": TPR_c,
        "max_acc_total": max_acc_total,
        "max_acc_last_stage": max_acc_last_stage
    }

    return output
