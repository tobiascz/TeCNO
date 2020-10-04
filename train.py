import configargparse
from pathlib import Path
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import (
    argparse_summary,
    get_class_by_path,
)
from utils.configargparse_arguments import build_configargparser

from datetime import datetime
logging.disable(logging.WARNING)

#SEED = 2334
#torch.manual_seed(SEED)
#np.random.seed(SEED)


def train(hparams, ModuleClass, ModelClass, DatasetClass, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    # load model
    model = ModelClass(hparams=hparams)
    # load dataset
    dataset = DatasetClass(hparams=hparams)
    # load module
    module = ModuleClass(hparams, model, dataset)

    # ------------------------
    # 3 INIT TRAINER --> continues training
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{hparams.output_path}/checkpoints/{hparams.name}",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='max',
        prefix=hparams.name)
    ### poly_logger has attribute with logging root for callback thingy for checkpoint
    progress_bar_log_per_epoch = 5
    biggest_data_split = max([len(dataset.data[k]) for k in dataset.data])
    progress_bar_refresh_rate_unrounded = biggest_data_split // progress_bar_log_per_epoch
    progress_bar_refresh_rate = round(
        progress_bar_refresh_rate_unrounded,
        -len(str(progress_bar_refresh_rate_unrounded)) + 1)
    progress_bar_refresh_rate = 0

    print(f"progress_bar_refresh_rate: {progress_bar_refresh_rate}")

    trainer = Trainer(
        gpus="0",
        logger=logger,
        fast_dev_run=hparams.fast_dev_run,
        overfit_pct=hparams.overfit_pct,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        train_percent_check=hparams.train_percent_check,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=True,
        gradient_clip_val=0,
        process_position=0,
        num_nodes=1,
        log_gpu_memory=None,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        track_grad_norm=-1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=1,
        val_percent_check=hparams.val_percent_check,
        test_percent_check=hparams.test_percent_check,
        val_check_interval=1.0,
        log_save_interval=100,
        row_log_interval=hparams.row_log_interval,
        distributed_backend=None,
        use_amp=False,
        print_nan_grads=False,
        weights_summary='full',
        weights_save_path=None,
        amp_level='O1',
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        truncated_bptt_steps=None,
        #resume_from_checkpoint=hparams.resume_from_checkpoint,
        resume_from_checkpoint=None,
    )
    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(module)
    print(
        f"Best: {checkpoint_callback.best} | monitor: {checkpoint_callback.monitor} | best_values: {checkpoint_callback.kth_value} | path: {checkpoint_callback.kth_best_model} |  "
    )

    if True:
        print(f"LOADING BEST: {checkpoint_callback.kth_best_model}")
        model_state_dict = module.model.state_dict()
        w_state_dict = torch.load(
            checkpoint_callback.kth_best_model)["state_dict"]
        for a in model_state_dict:
            model_state_dict[a] = w_state_dict["model." + a]
        module.model.load_state_dict(model_state_dict)

        trainer.test(module)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # each LightningModule defines arguments relevant to it
    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)
    # ------------------------
    # LOAD MODEL
    # ------------------------
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)
    # ------------------------
    # LOAD DATASET
    # ------------------------
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)
    # ------------------------
    #  PRINT PARAMS & INIT LOGGER
    # ------------------------
    hparams = parser.parse_args()
    # seutp logging
    exp_name = (hparams.module.split(".")[-1] + "_" +
                hparams.dataset.split(".")[-1] + "_" +
                hparams.model.replace(".", "_"))

    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
    hparams.name = date_str + exp_name
    hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    tb_logger = TensorBoardLogger(hparams.output_path, name='tb')
    print('Output path: ', hparams.output_path)
    loggers = [tb_logger]

    argparse_summary(hparams, parser)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    train(hparams, ModuleClass, ModelClass, DatasetClass, loggers)
