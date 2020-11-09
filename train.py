import configargparse
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

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
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='max',
        prefix=hparams.name,
        filename=f'{{epoch}}-{{{hparams.early_stopping_metric}:.2f}}'
    )
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=3,
        mode='max')


    trainer = Trainer(
        gpus=hparams.gpus,
        logger=logger,
        fast_dev_run=hparams.fast_dev_run,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        callbacks=[early_stop_callback],
        weights_summary='full',
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps
    )
    # ------------------------
    # 4 START TRAINING
    # ------------------------

    trainer.fit(module)
    print(
        f"Best: {checkpoint_callback.best_model_score} | monitor: {checkpoint_callback.monitor} | path: {checkpoint_callback.best_model_path}"
        f"\nTesting..."
    )
    trainer.test(ckpt_path=checkpoint_callback.best_model_path)



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
    # setup logging

    exp_name = (hparams.module.split(".")[-1] + "_" + hparams.dataset.split(".")[-1] + "_" + hparams.model.replace(".",
        "_"))
    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
    hparams.name = date_str + exp_name
    hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    tb_logger = TensorBoardLogger(hparams.output_path, name='tb')
    wandb_logger = WandbLogger(name = hparams.name, project="tecno")

    argparse_summary(hparams, parser)
    print('Output path: ', hparams.output_path)


    loggers = [tb_logger, wandb_logger]
    # ---------------------
    # RUN TRAINING
    # ---------------------
    train(hparams, ModuleClass, ModelClass, DatasetClass, loggers)
